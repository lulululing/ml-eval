# 作业3评估工程

本项目用于完成两项评测：
- 任务一：LLM 数学推理评测（GSM8K），随机抽取 Test 集 50 条，输出逐步推理与最终答案，计算准确率。
- 任务二：MLLM 多模态科学问答（ScienceQA，Image-only），随机抽取 Test 集 50 条，输入图片+题干+选项，从模型输出选择 A/B/C/D，计算准确率。

支持两类后端：
- HF Transformers（推荐模型：Qwen2.5-7B-Instruct，Qwen2-VL-7B-Instruct 或 LLaVA-v1.5-7b），可启用4-bit量化（需GPU）
- llama.cpp / llava.cpp（CPU可运行的GGUF量化模型），满足在无GPU机器上的运行需求

> 注意：在Windows无GPU环境下，建议使用 llama.cpp/llava.cpp 的 GGUF 量化模型运行；使用HF 4-bit 量化通常需要CUDA支持。

## 快速开始（Windows PowerShell）

### 1. 创建并安装依赖（HF后端）
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt
```

### 2. 评测 GSM8K（LLM）
- 选用HF后端（需GPU）示例：
```powershell
python src/gsm8k_eval.py --backend hf --model "Qwen/Qwen2.5-7B-Instruct" --quantization 4bit --sample 50 --out results/gsm8k_hf.jsonl
```
- 选用llama.cpp后端（CPU，GGUF）：先下载对应7B的GGUF模型文件（例如 Qwen2.5-7B-Instruct 的兼容GGUF）到 `models/llm.gguf`
```powershell
python src/gsm8k_eval.py --backend llama_cpp --gguf models/llm.gguf --sample 50 --out results/gsm8k_llama_cpp.jsonl
```

### 3. 评测 ScienceQA（MLLM）
- HF后端（需GPU）：
```powershell
python src/scienceqa_eval.py --backend hf --model "Qwen/Qwen2-VL-7B-Instruct" --sample 50 --out results/scienceqa_hf.jsonl
```
- llava.cpp后端（CPU）：下载 7B 的 LLaVA GGUF 模型与对应图像编码文件，配置 `--gguf` 路径
```powershell
python src/scienceqa_eval.py --backend llava_cpp --gguf models/llava-7b.gguf --sample 50 --out results/scienceqa_llava_cpp.jsonl
```

### 4. 输出与准确率
脚本会将逐条样本的模型输出（含推理过程/选项判断）写入 JSONL 文件，并在末尾打印准确率统计；同时在 `results/metrics.json` 汇总。

## 数据下载
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- ScienceQA: https://huggingface.co/datasets/derek-thomas/ScienceQA

首次运行会自动通过 `datasets` 库缓存数据；ScienceQA 图像样本会自动下载对应图像文件。

## 模型结构说明（报告需要）
- Qwen2.5-7B-Instruct（LLM）：Transformer 解码器架构，使用自注意力与前馈网络，经过指令微调；通过提示词让模型进行链式思维（CoT）推理。
- Qwen2-VL-7B-Instruct（MLLM）：视觉编码器（例如 ViT/CLIP 类）+ 文本解码器（Qwen2系），通过多模态投影将图像特征与文本token融合，生成答案。
- LLaVA-v1.5-7b：CLIP视觉编码器 + LLaMA 文本解码器，通过视觉-文本对齐训练实现多模态问答。

## 4-bit量化说明
- HF后端的 `bitsandbytes` 4-bit 量化通常需要 GPU（CUDA），可降低显存使用。
- 在CPU上建议改用 `llama.cpp`/`llava.cpp` 的 GGUF（如 q4_0、q4_k_m）量化模型，通过 `llama-cpp-python` 或独立可执行程序调用。

## 报告要点（建议）
- 模型结构简述（引用模型原文/官方说明）
- Pipeline：数据加载→提示构造→模型生成→答案解析→准确率计算
- 资源配置：GPU/CPU、量化设置、推理参数（温度、最大token）
- 结果：整体准确率、案例展示（含推理过程）、错误分析

## 项目结构
```
src/
  gsm8k_eval.py
  scienceqa_eval.py
  utils/
    eval.py
    modeling_llm.py
    modeling_mllm.py
results/
models/
```
