import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import swanlab

os.environ["SWANLAB_PROJECT"] = "qwen3-sft-network"

PROMPT = "你是一位资深的计算机网络专家教授，具有丰富的教学和研究经验。你需要针对用户提出的计算机网络相关问题，先进行一步步的思考，然后给出准确、详细且易懂的回答。"
MAX_LENGTH = 2048

swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    "finetune_method": "LoRA (r=64)"  # 可改为 "QLoRA-4bit" 如果开启量化
})

# ====================== 数据格式转换函数 ======================
def dataset_jsonl_transfer(origin_path, new_path):
    messages = []
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            input_text = data["question"]
            output = f"<think>{data['think']}</think>\n{data['answer']}"
            message = {
                "instruction": PROMPT,
                "input": input_text,
                "output": output,
            }
            messages.append(message)

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# ====================== 数据预处理函数 ======================
def process_func(example):
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ====================== 推理函数 ======================
def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=MAX_LENGTH)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# ====================== 下载并加载模型 ======================
model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="./Models", revision="master")

# ------------------- 可选：开启 4bit 量化（QLoRA，极致省显存） -------------------
# 如果你显存只有 16GB 左右，强烈建议打开下面这几行（把注释去掉）
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "./Models/Qwen/Qwen3-1.7B",
#     quantization_config=quantization_config,
#     device_map="auto",
#     trust_remote_code=True
# )
# model = prepare_model_for_kbit_training(model)  # QLoRA 必须加这句

# ------------------- 不开量化时的普通加载（LoRA） -------------------
model = AutoModelForCausalLM.from_pretrained(
    "./Models/Qwen/Qwen3-1.7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("./Models/Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)

# ====================== 添加 LoRA 配置（关键部分） ======================
peft_config = LoraConfig(
    r=64,                                      # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],  # Qwen3 的所有线性层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# 打印可训练参数数量（应该只有几千万，约 1~2% 的原参数）
model.print_trainable_parameters()

# 开启梯度检查点（LoRA 下可以放心开，显存更省）
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# ====================== 数据加载与处理 ======================
train_dataset_path = "datasets/train.jsonl"
test_dataset_path = "datasets/val.jsonl"
train_jsonl_new_path = "datasets/train_format.jsonl"
test_jsonl_new_path = "datasets/val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

# ====================== 训练参数 ======================
args = TrainingArguments(
    output_dir="./output/Qwen3-1.7B-network-lora-2",
    per_device_train_batch_size=4,           # LoRA 下可以把 batch 调大一点
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,           # 有效 batch_size = 4*2 = 8
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=3,                      # LoRA 可以稍微多训一点，3 epoch 很常见
    learning_rate=2e-4,                      # LoRA 推荐 1e-4 ~ 5e-4
    save_steps=200,
    save_total_limit=3,
    fp16=False,                              # 用 bfloat16 的话保持默认
    bf16=True,                               # 如果显卡支持 bfloat16（A100/RTX4090等）
    gradient_checkpointing=True,
    optim="paged_adamw_8bit" if getattr(model, "is_quantized", False) else "adamw_torch",
    report_to="swanlab",
    run_name="qwen3-1.7B-network-lora",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# ====================== 测试前3条 ======================
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]
test_text_list = []

for _, row in test_df.iterrows():
    messages = [
        {"role": "system", "content": row['instruction']},
        {"role": "user", "content": row['input']}
    ]
    response = predict(messages, model, tokenizer)
    response_text = f"""
    Question: {row['input']}

    LLM: {response}
    """
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})
swanlab.finish()