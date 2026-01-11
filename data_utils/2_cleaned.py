import json
import re
import nltk
from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  

INPUT_JSONL = "../datasets/extracted_text.jsonl"          # 上一步生成的
OUTPUT_DATASET = "../datasets/network_dataset.jsonl"       # 最终初步数据集

# 根据实际情况设置：从第几页开始是正文（你可以先运行查看内容）
START_PAGE = 15  # 常见正文从第10-20页开始，建议先设小一点，手动调整

def clean_content(text):
    """深度清洗单页内容"""
    # 移除常见噪声
    text = re.sub(r'第\s*\d+\s*页.*', '', text)
    text = re.sub(r'^\s*\d+\s*计算机网络.*第\d+版.*', '', text)
    text = re.sub(r'清\s*华\s*大\s*学\s*出\s*版\s*社', '', text)
    text = re.sub(r'Andrew S\. Tanenbaum.*', '', text)
    text = re.sub(r'前\s*言|前言|目\s*录|目录', '', text)
    text = re.sub(r'第\s*\d+\s*章.*', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+\n', '\n', text)
    return text.strip()

# 读取所有页面
pages = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            pages.append(json.loads(line))

print(f"总共 {len(pages)} 页")

# 过滤并清洗
cleaned_paragraphs = []
current_para = ""

for item in pages:
    page_num = item["page"]
    if page_num < START_PAGE:
        continue  # 跳过前言、目录等
    
    content = clean_content(item["content"])
    if len(content) < 50:
        continue
    
    # 简单合并：把连续页面内容连起来，形成段落
    sentences = nltk.sent_tokenize(content)
    for sent in sentences:
        if len(current_para + sent) > 800:
            if len(current_para) > 100:
                cleaned_paragraphs.append(current_para.strip())
            current_para = sent + " "
        else:
            current_para += sent + " "
    
    # 每页结束时强制切一段（避免跨章太长）
    if current_para.strip():
        cleaned_paragraphs.append(current_para.strip())
        current_para = ""

# 最后一段
if current_para.strip():
    cleaned_paragraphs.append(current_para.strip())

print(f"生成 {len(cleaned_paragraphs)} 个有效段落")

# 生成初步数据集（规则版）
dataset = []
templates = [
    "什么是{}？",
    "{}的原理是什么？",
    "请解释{}的过程。",
    "{}协议的工作机制是怎样的？",
    "在计算机网络中，{}的作用是什么？",
    "请逐步说明{}的步骤。",
    "{}和{}有什么区别？"
]

for i, para in enumerate(tqdm(cleaned_paragraphs[:1500])):  # 先取前1500段
    words = re.findall(r'[\u4e00-\u9fa5A-Za-z0-9]+', para)
    keywords = [w for w in words if len(w) >= 2 and w not in ['计算机', '网络', '的', '是']]
    if not keywords:
        continue
    
    key1 = keywords[0]
    key2 = keywords[min(3, len(keywords)-1)] if len(keywords) > 3 else "TCP"
    
    template = templates[i % len(templates)]
    if "{}和{}" in template:
        question = template.format(key1, key2)
    else:
        question = template.format(key1)
    
    think = f"用户询问计算机网络中的“{key1}”相关知识。我需要根据教材内容，准确回忆定义、原理和工作机制，然后结构化、逐步地进行解释。"
    answer = para
    
    dataset.append({
        "question": question,
        "think": think,
        "answer": answer
    })

# 保存为标准 JSONL
with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"初步数据集生成完成！共 {len(dataset)} 条 → {OUTPUT_DATASET}")
print("\n前3条预览：")
for i, item in enumerate(dataset[:3]):
    print(f"\n第{i+1}条")
    print("Q:", item["question"])
    print("Think:", item["think"])
    print("A:", item["answer"][:150] + "...")