import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import evaluate
import re
from tqdm import tqdm
import argparse

# ------------------ 命令行参数 ------------------
parser = argparse.ArgumentParser(description="Qwen3 模型验证脚本 - 支持 LoRA 与全参数微调")
parser.add_argument("--model_type", type=str, default="lora",
                    choices=["lora", "full"],
                    help="模型类型：'lora' 或 'full'")
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="检查点路径")
parser.add_argument("--test_file", type=str, default="val_format.jsonl",
                    help="测试集路径（格式化后的 jsonl）")
parser.add_argument("--max_samples", type=int, default=1024,
                    help="评估的最大样本数")
parser.add_argument("--max_new_tokens", type=int, default=1024,
                    help="最大生成长度")
parser.add_argument("--system_prompt", type=str, default=None,
                    help="自定义系统提示词")
parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="用于 G-Eval 打分的大模型（建议 7B 以上）")

args = parser.parse_args()

# 系统提示词（与训练时保持一致）
DEFAULT_SYSTEM_PROMPT = (
    "你是一位资深的计算机网络专家教授，具有丰富的教学和研究经验。"
    "你需要针对用户提出的计算机网络相关问题，先进行一步步的思考，然后给出准确、详细且易懂的回答。"
)

SYSTEM_PROMPT = args.system_prompt if args.system_prompt else DEFAULT_SYSTEM_PROMPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 加载被评估模型与 Tokenizer ====================
print("正在加载被评估模型，请稍等...")

base_model_name = "Qwen/Qwen3-1.7B"  # ← 请根据实际情况修改

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    use_fast=False,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

if args.model_type.lower() == "lora":
    print("加载 LoRA 微调模型...")
    peft_config = PeftConfig.from_pretrained(args.checkpoint_path)
    model = PeftModel.from_pretrained(model, args.checkpoint_path, device_map="auto")
else:
    print("加载全参数微调模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

model.eval()
print(f"模型加载完成！类型: {args.model_type} | 设备: {DEVICE}\n")

# ==================== 加载 G-Eval 评委模型 ====================
print("正在加载 G-Eval 评委模型（可能需要一些时间）...")
judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
judge_model = AutoModelForCausalLM.from_pretrained(
    args.judge_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
judge_pipe = pipeline(
    "text-generation",
    model=judge_model,
    tokenizer=judge_tokenizer,
    device_map="auto"
)
print("评委模型加载完成\n")

# ==================== 加载测试集 ====================
with open(args.test_file, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

print(f"加载了 {len(test_data)} 条测试数据，将评估前 {args.max_samples} 条\n")

# ==================== 加载评估指标 ====================
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
chrf = evaluate.load("chrf")  # 用于计算 ChrF++

# ==================== G-Eval 提示模板（可根据需求调整） ====================
GEVAL_PROMPT = """\
你是一位专业的评估专家。
请根据以下标准，对模型生成的回答进行评分（满分 10 分）：

评分标准：
1. 准确性：回答是否正确、没有事实错误（权重最高）
2. 完整性：是否涵盖了问题的主要方面
3. 逻辑性与思考过程：是否有清晰的 <think>...</think> 思考步骤
4. 可读性与专业性：语言是否专业、清晰、易懂

问题：{question}

参考答案：{reference}

模型生成回答：{response}

请直接给出一个 0~10 的整数分数（只输出数字，不要解释），越高代表整体质量越好。
"""

# ==================== 开始评估 ====================
think_count = 0
results = []
eval_samples = min(args.max_samples, len(test_data))

for item in tqdm(test_data[:eval_samples], desc="评估进度", unit="条"):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["input"]}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # 是否包含思考过程
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    if has_think:
        think_count += 1
    
    reference = item["output"]
    question = item["input"]
    
    # ChrF++ (word_order=2 即 ChrF++)
    chrf_result = chrf.compute(
        predictions=[response],
        references=[[reference]],
        char_order=6,
        word_order=2,
        beta=2
    )
    chrf_pp = chrf_result['score'] / 100.0  # 转为 0~1 范围，方便比较
    
    # ROUGE-L
    rouge_result = rouge.compute(predictions=[response], references=[reference])
    rouge_l = rouge_result["rougeL"]
    
    # BERTScore
    bert_result = bertscore.compute(predictions=[response], references=[reference], lang="zh")
    bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])
    
    # G-Eval 分数（使用强模型打分）
    geval_input = GEVAL_PROMPT.format(
        question=question,
        reference=reference,
        response=response
    )
    
    try:
        judge_output = judge_pipe(
            geval_input,
            max_new_tokens=32,
            temperature=0.1,
            do_sample=False,
            num_return_sequences=1
        )[0]['generated_text'].strip()
        
        # 尝试提取数字分数
        geval_score = 0.0
        for token in judge_output.split():
            try:
                score = float(token)
                if 0 <= score <= 10:
                    geval_score = score
                    break
            except:
                pass
        geval_score = geval_score / 10.0  # 转为 0~1
    except:
        geval_score = 0.0  # 防错处理

    results.append({
        "question": question,
        "response": response,
        "has_think": has_think,
        "chrf_pp": chrf_pp,
        "rougeL": rouge_l,
        "bertscore": bert_f1,
        "geval": geval_score
    })

# ==================== 输出结果 ====================
if results:
    think_rate = think_count / len(results) * 100
    avg_chrf = sum(r["chrf_pp"] for r in results) / len(results)
    avg_rouge = sum(r["rougeL"] for r in results) / len(results)
    avg_bert = sum(r["bertscore"] for r in results) / len(results)
    avg_geval = sum(r["geval"] for r in results) / len(results)
else:
    think_rate = avg_chrf = avg_rouge = avg_bert = avg_geval = 0.0

print("\n" + "="*80)
print(f"评估完成！（样本数：{len(results)}）")
print(f"模型类型       : {args.model_type.upper()}")
print(f"检查点路径     : {args.checkpoint_path}")
print(f"Think Rate（思考率） : {think_rate:6.2f}%  ({think_count}/{len(results)})")
print(f"Avg ChrF++           : {avg_chrf:.4f}")
print(f"Average ROUGE-L      : {avg_rouge:.4f}")
print(f"Average BERTScore    : {avg_bert:.4f}")
print(f"Average G-Eval (0-1) : {avg_geval:.4f}")
print("="*80)

if avg_geval >= 0.85 and avg_chrf > 0.75:
    print("🎉 模型表现非常优秀！可以考虑正式部署！")
elif avg_geval >= 0.70 and avg_chrf > 0.60:
    print("👍 模型表现良好，基本可用，可继续观察或轻微优化")
else:
    print("⚠ 模型表现一般，建议检查数据质量、提示词或训练设置")

"""
验证 LoRA 模型python eval_model.py --model_type lora --checkpoint_path output/Qwen3-1.7B-network-lora/checkpoint-321 --test_file datasets/val_format.jsonl --max_samples 30"""
"""
验证全参数微调模型
python eval_model.py --model_type full --checkpoint_path output/Qwen3-1.7B-network/checkpoint-426 --test_file datasets/val_format.jsonl --max_samples 30
"""

