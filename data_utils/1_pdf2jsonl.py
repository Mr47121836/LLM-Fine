import pdfplumber
import json

def extract_pdf_text_to_jsonl(pdf_path, output_jsonl="extracted_text.jsonl"):
    """
    从 PDF 提取文本，直接保存为 JSONL 格式（每行一个 {"page": x, "content": "..."}）
    """
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"PDF 共 {total_pages} 页，开始提取...")

        with open(output_jsonl, "w", encoding="utf-8") as f:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:  # 忽略空白页或提取失败的页
                    record = {
                        "page": page_num,
                        "content": text.strip()
                    }
                    # 每行写入一个 JSON 对象，后加换行
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    # 可选：记录空白页，便于调试
                    print(f"第 {page_num} 页无文本内容，已跳过")

    print(f"提取完成！共处理 {total_pages} 页")
    print(f"已保存为 JSONL 格式：{output_jsonl}")
    print("   → 每行一个 JSON 对象，可直接用于后续清洗和数据集生成")

# ==================== 使用部分 ====================
pdf_path = "../datasets/computer_networking.pdf"          # ←←← 修改为你的 PDF 文件实际路径
output_jsonl = "../datasets/extracted_text.jsonl"          # 输出文件名，可自定义

extract_pdf_text_to_jsonl(pdf_path, output_jsonl)