import argparse
import random
import json
import os
from typing import Tuple

from datasets import load_dataset
from PIL import Image

from src.utils.modeling_mllm import get_mllm, BACKEND_HF, BACKEND_LLAVA_CPP
from src.utils.eval import extract_choice, accuracy, write_jsonl
import sys
import time

PROMPT_SUFFIX = "Please choose the correct option (A/B/C/D)."


def load_image(img_data):
    """安全地加载图片，兼容多种数据格式"""
    if img_data is None:
        return None
    
    # 如果已经是PIL Image对象
    if isinstance(img_data, Image.Image):
        return img_data.convert('RGB')
    
    # 如果是文件路径字符串
    if isinstance(img_data, str):
        return Image.open(img_data).convert('RGB')
    
    # 如果是其他格式，尝试直接转换
    try:
        if hasattr(img_data, 'convert'):
            return img_data.convert('RGB')
        return img_data
    except Exception as e:
        print(f"Warning: Could not convert image: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=[BACKEND_HF, BACKEND_LLAVA_CPP], default=BACKEND_HF)
    ap.add_argument('--model', type=str, help='HF MLLM model name (e.g., Qwen/Qwen2-VL-7B-Instruct)')
    ap.add_argument('--gguf', type=str, help='Path to LLaVA GGUF if using llava.cpp backend')
    ap.add_argument('--sample', type=int, default=50)
    ap.add_argument('--out', type=str, default='results/scienceqa.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # 确保results目录存在
    os.makedirs('results', exist_ok=True)

    random.seed(args.seed)
    print("Loading ScienceQA dataset...")
    try:
        ds = load_dataset('derek-thomas/ScienceQA', 'default')['test']
    except Exception as e:
        print(f"[FATAL] Failed to load ScienceQA dataset: {e}", file=sys.stderr)
        sys.exit(1)
    # 只选择包含图片的样本
    print("Filtering samples with images...")
    image_idxs = [i for i in range(len(ds)) if ds[i].get('image') is not None]
    print(f"Found {len(image_idxs)} samples with images")
    
    if len(image_idxs) < args.sample:
        print(f"Warning: Only {len(image_idxs)} samples available, using all of them")
        idxs = image_idxs
    else:
        idxs = random.sample(image_idxs, args.sample)

    print(f"Initializing MLLM with backend={args.backend}...")
    try:
        gen = get_mllm(args.backend, model_name=args.model, gguf_path=args.gguf)
    except Exception as e:
        print(f"[FATAL] Failed to load MLLM: {e}", file=sys.stderr)
        sys.exit(1)

    rows = []
    preds = []
    labels = []

    print(f"Evaluating {len(idxs)} samples...")
    for idx, i in enumerate(idxs):
        item = ds[i]
        img_data = item['image']
        question = item['question']
        
        # 确保choices是列表且至少有4个选项
        choices_list = item.get('choices', [])
        if len(choices_list) < 4:
            print(f"Skipping question {i}: insufficient choices")
            continue
            
        choices = (choices_list[0], choices_list[1], choices_list[2], choices_list[3])
        label_letter = item['answer']  # 应该是A/B/C/D

        print(f"[{idx+1}/{len(idxs)}] Processing question {i}...")
        
        try:
            # 安全加载图片
            image = load_image(img_data)
            if image is None:
                print(f"Skipping question {i}: could not load image")
                continue
            
            # 生成回答
            out = gen(image, question, choices, max_tokens=256, temperature=0.2)
            choice = extract_choice(out)

            # 保存结果
            rows.append({
                'id': i,
                'question': question,
                'choices': {'A': choices[0], 'B': choices[1], 'C': choices[2], 'D': choices[3]},
                'model_output': out,
                'pred': choice,
                'label': label_letter
            })

            preds.append(choice)
            labels.append(label_letter)
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue

    if len(preds) == 0:
        print("Error: No samples were successfully processed!")
        return

    acc = accuracy(preds, labels)
    print(f"\n{'='*50}")
    print(f"ScienceQA (image-only) accuracy on {len(preds)} samples: {acc:.3f} ({acc*100:.1f}%)")
    print(f"{'='*50}\n")

    # 保存详细结果
    write_jsonl(args.out, rows)
    print(f"Detailed results saved to {args.out}")

    # 读取或创建metrics文件（不覆盖已有的指标）
    metrics_path = 'results/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    # 更新ScienceQA的metrics
    metrics['scienceqa'] = {
        'samples': len(preds), 
        'accuracy': acc,
        'correct': sum(p == l for p, l in zip(preds, labels)),
        'total': len(labels)
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics updated in {metrics_path}")


if __name__ == '__main__':
    main()
