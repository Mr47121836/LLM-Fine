import json
import dashscope
from dashscope import Generation
from tqdm import tqdm
import time
import re
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

# 配置日志（线程安全）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Key 设置（你当前硬编码的方式）
dashscope.api_key = "sk-d1a3e8640ca84879bd9e583111cfeaef"
logger.info("API Key 已成功加载")

# 配置
INPUT_JSONL = "../datasets/network_dataset.jsonl"
OUTPUT_JSONL = "../datasets/enhanced_network_dataset.jsonl"
CHECKPOINT_FILE = "../datasets/checkpoint.txt"
MODEL = "qwen-max"
MAX_SAMPLES = None
MAX_WORKERS = 6  # 推荐 4~8，根据你的网络和API限额调整，太多可能被限流
REQUEST_DELAY = 0.3  # 每个请求后基础延时（秒），多线程下可适当降低

# 线程安全的文件写入锁
write_lock = Lock()
# 线程局部存储（避免线程间干扰）
thread_local = threading.local()

def get_response_content():
    """线程安全的获取 Generation.call 响应"""
    if not hasattr(thread_local, "retry_count"):
        thread_local.retry_count = 0

    prompt_template = """你是一位计算机网络专家。根据以下教材原文，生成一条高质量的微调数据示例。

要求（必须严格遵守）：
- question: 自然、口语化的学生问题（例如："老师，TCP 三次握手是怎么回事？"）
- think: 结构化的逐步推理过程
- answer: 专业易懂的完整回答，使用分点或编号

原文：
{raw_answer}

示例输出：
{{
  "question": "老师，TCP 三次握手为什么不能用两次？",
  "think": "学生在问TCP连接建立的安全性。我需要先回忆三次握手的目的是防止旧连接干扰、确认双向通信能力，然后解释两次握手无法实现同步确认，最后总结三次握手的必要性。",
  "answer": "TCP 使用三次握手建立连接，主要原因如下：\\n1. 防止历史连接干扰...\\n2. 实现双向确认...\\n3. 同步序列号..."
}}

请严格只输出一个纯 JSON 对象，不要有任何说明文字、代码块标记、换行说明等：
{{"question": "xxx", "think": "xxx", "answer": "xxx"}}"""

    def call_api(raw_answer):
        prompt = prompt_template.format(raw_answer=raw_answer)
        messages = [
            {'role': 'system', 'content': '你是一个严谨的JSON生成助手，只输出纯JSON。'},
            {'role': 'user', 'content': prompt}
        ]

        for attempt in range(3):
            try:
                response = Generation.call(
                    model=MODEL,
                    messages=messages,
                    result_format='message',
                    temperature=0.5,
                    top_p=0.8
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content.strip()
                else:
                    logger.warning(f"线程{threading.current_thread().name} API错误: {response.message}，重试 {attempt+1}/3")
            except Exception as e:
                logger.error(f"线程{threading.current_thread().name} 调用异常: {e}，重试 {attempt+1}/3")
            time.sleep(2 ** attempt)
        return None

    return call_api

def parse_json_content(content):
    """健壮的JSON解析"""
    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = json_str.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON解析失败: {str(e)[:100]}...")
    
    return {
        "question": "解析失败",
        "think": "模型输出格式错误",
        "answer": content[:800]
    }

def process_item(item, index, total):
    """处理单条数据"""
    raw_answer = item.get("answer", "").strip()
    if len(raw_answer) < 50:
        return None

    content = get_response_content()(raw_answer)
    if not content:
        logger.error(f"第 {index+1}/{total} 条 API 调用失败")
        return None

    result = parse_json_content(content)
    if result:
        # 限流（每个线程独立控制）
        time.sleep(REQUEST_DELAY)
        return result
    
    return None

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(index))

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def main():
    # 读取数据
    data = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if item.get("answer", "").strip():
                        data.append(item)
                except:
                    pass

    if MAX_SAMPLES:
        data = data[:MAX_SAMPLES]

    total = len(data)
    logger.info(f"总共加载 {total} 条数据，准备使用 {MAX_WORKERS} 个线程增强")

    start_index = load_checkpoint()
    data_to_process = data[start_index:]

    enhanced_count = start_index
    results = [None] * len(data_to_process)  # 预占位保持顺序

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_item, item, start_index + i, total): i
            for i, item in enumerate(data_to_process)
        }

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="增强数据", initial=0):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result:
                    results[idx] = result
                    enhanced_count += 1
                    # 每10条保存一次检查点
                    if enhanced_count % 10 == 0:
                        save_checkpoint(enhanced_count + start_index - (len(data_to_process) - len([r for r in results if r is not None])))
            except Exception as e:
                logger.error(f"线程任务异常: {e}")

    # 写入文件（保持原始顺序）
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        with write_lock:
            # 先写已完成的（从0到start_index）
            if start_index > 0:
                with open(OUTPUT_JSONL, "r", encoding="utf-8") as existing:
                    for line in existing:
                        f.write(line)
            # 再写新增强的
            for result in results:
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    logger.info(f"多线程增强完成！共成功增强约 {enhanced_count} 条 → {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()