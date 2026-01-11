import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

# ==================== 配置区域 ====================
# 修改这里：你的 LoRA 训练输出目录（Trainer 保存的 checkpoint）
# 通常是最后一个或最好的 checkpoint，比如：
#   ./output/Qwen3-1.7B-network-lora/checkpoint-600
#   或者直接用最终合并后的目录（如果你合并了的话）
PEFT_MODEL_PATH = "output/Qwen3-1.7B-network-lora/checkpoint-321"   # 

# 系统提示词（和训练时完全一致！非常重要）
SYSTEM_PROMPT = "你是一位资深的计算机网络专家教授，具有丰富的教学和研究经验。你需要针对用户提出的计算机网络相关问题，先进行一步步的思考，然后给出准确、详细且易懂的回答。"

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 加载模型和Tokenizer ====================
print("正在加载模型，请稍等...")

# 先加载 PEFT 配置
peft_config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)

# 加载基础模型（Qwen3-1.7B）
base_model_name = peft_config.base_model_name_or_path
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,   # 或 torch.float16，根据你的显卡
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)
model.eval()  # 推理模式

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False, trust_remote_code=True)

print(f"模型加载完成！使用设备: {DEVICE}")
print("可以开始提问了（输入 'quit' 或 '退出' 结束对话）\n")

# ==================== 聊天循环 ====================
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    user_input = input("你（用户）: ").strip()
    
    if user_input.lower() in ["quit", "退出", "exit"]:
        print("再见！")
        break
    
    if user_input == "":
        print("请输入问题～")
        continue

    # 添加用户消息
    messages.append({"role": "user", "content": user_input})

    # 构造输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,           # 根据需要调整
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码，只取新生成的部分
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"\n网络专家教授:\n{response}\n")
    
    # 添加助手回复到历史
    messages.append({"role": "assistant", "content": response})