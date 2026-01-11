

# LLM-Fine

<div align="center">

一个专注于**大语言模型（LLM）参数高效微调（PEFT）**的实用项目集合

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-supported-brightgreen)](https://github.com/huggingface/peft)

</div>

## 项目介绍

**LLM-Fine** 是一个用于大语言模型微调的代码仓库，旨在提供清晰、可复现的微调流程，支持多种**参数高效微调**方法（PEFT），方便研究者和开发者快速上手 LLM 的 SFT（监督微调）、指令微调、LoRA、QLoRA 等实验。

目前支持：
- LoRA / QLoRA 等主流 PEFT 方法
- 标准 SFT 训练流程
- 模型推理与评估脚本
- 常见中文/英文指令数据集处理

## 目前支持的功能

- 数据预处理与格式化（Alpaca/ShareGPT 等格式）
- LoRA / QLoRA 微调（基于 `peft` 库）
- RoLA（可能为自定义或变种方法）训练支持
- 训练后模型合并、推理与评估
- 支持 4bit/8bit 量化加载（节省显存）

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Mr47121836/LLM-Fine.git
cd LLM-Fine
```

### 2. 安装依赖

```bash
pip install -r requirments.txt
```

**推荐使用 conda 环境**：

```bash
conda create -n llm-fine python=3.12
conda activate llm-fine
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. 准备数据

将你的数据集放在 `datasets/` 目录下，支持常见格式：

- Alpaca 格式 JSON
- ShareGPT 格式 JSON
- 纯文本对话格式

示例数据结构（Alpaca）：

```json
[
  {
    "instruction": "写一首关于秋天的诗",
    "input": "",
    "output": "落叶萧萧下..."
  }
]
```

### 4. 开始训练（以 LoRA 为例）

```bash
# 标准 LoRA 微调
python train.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --data_path "datasets/your_dataset.json" \
    --output_dir "output/lora-llama2-7b" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3
```

**RoLA 训练**（如果仓库中有此方法）：

```bash
python RoLA_train.py --config configs/rola_config.yaml
```

### 5. 推理 & 评估

```bash
# 单条推理
python inference.py --model_path output/lora-llama2-7b --prompt "你好，今天心情如何？"

# 批量评估
python eval_model.py --model_path output/lora-llama2-7b --test_data datasets/test.json
```

## 仓库结构说明

```
LLM-Fine/
├── data_utils/          # 数据处理工具
├── datasets/            # 示例/存放数据集
├── peft/                # PEFT 相关自定义代码（如果有）
├── train.py             # 标准 LoRA/QLoRA 训练脚本
├── RoLA_train.py        # RoLA 训练脚本（待完善说明）
├── inference.py         # 模型推理脚本
├── eval_model.py        # 模型评估脚本
├── LLMM-Fine.ipynb      # Jupyter Notebook 实验笔记本（推荐用于调试）
├── requirements.txt     # 依赖列表（建议改名为 requirements.txt）
└── README.md
```

## 待完善事项（欢迎 PR！）

- 更详细的配置文件示例（yaml/json）
- 支持更多基座模型（Qwen、DeepSeek、Yi、GLM 等）
- 添加更多评估指标（BLEU、ROUGE、Perplexity、Reward Model 分数等）
- 增加 Web UI 推理 Demo（Gradio / Streamlit）
- 更完善的文档和使用教程

## 致谢

感谢以下优秀开源项目：

- [huggingface/peft](https://github.com/huggingface/peft)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [unsloth](https://github.com/unslothai/unsloth) （如果使用了 2x 速度加速）
- 所有开源数据集和模型贡献者

## Star & Fork

如果你觉得这个项目对你有帮助，欢迎点个 ⭐ 支持一下～

也欢迎提交 Issue 或 Pull Request，一起把 LLM 微调玩得更好！

<div align="right">
最后更新：2026年1月
</div>
```
