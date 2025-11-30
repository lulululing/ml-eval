import argparse
import random
import json
import os
from typing import Tuple

from datasets import load_dataset
from PIL import Image

from src.utils.modeling_mllm import get_mllm, BACKEND_HF, BACKEND_LLAVA_CPP
from src.utils.eval import extract_choice, accuracy, write_jsonl

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


def filter_valid_samples(ds):
    """筛选有效样本：有图片且选项数>=2"""
    valid_idxs = []
    for i in range(len(ds)):
        item = ds[i]
        # 检查是否有图片
        if item.get('image') is None:
            continue
        # 检查choices数量
        choices_list = item.get('choices', [])
        if len(choices_list) >= 2:  # 至少2个选项就可以
            valid_idxs.append(i)
    return valid_idxs


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
    ds = load_dataset('derek-thomas/ScienceQA', 'default')['test']
    
    # 筛选有效样本
    print("Filtering valid samples (with images and sufficient choices)...")
    valid_idxs = filter_valid_samples(ds)
    print(f"Found {len(valid_idxs)} valid samples")
    
    if len(valid_idxs) == 0:
        print("Error: No valid samples found!")
        return
    
    if len(valid_idxs) < args.sample:
        print(f"Warning: Only {len(valid_idxs)} samples available, using all of them")
        idxs = valid_idxs
    else:
        idxs = random.sample(valid_idxs, args.sample)

    print(f"Initializing MLLM with backend={args.backend}...")
    gen = get_mllm(args.backend, model_name=args.model, gguf_path=args.gguf)

    rows = []
    preds = []
    labels = []
    skipped = 0

    print(f"Evaluating {len(idxs)} samples...")
    for idx, i in enumerate(idxs):
        item = ds[i]
        img_data = item['image']
        question = item['question']
        
        # 获取choices
        choices_list = item.get('choices', [])
        num_choices = len(choices_list)
        
        # 补齐选项到4个（如果不足）
        while len(choices_list) < 4:
            choices_list.append(f"Option {len(choices_list) + 1}")
        
        choices = (choices_list[0], choices_list[1], choices_list[2], choices_list[3])
        
        # 获取正确答案
        answer_idx = item.get('answer')
        if isinstance(answer_idx, int):
            label_letter = ['A', 'B', 'C', 'D'][answer_idx]
        else:
            label_letter = str(answer_idx).upper()
        
        # 如果答案超出实际选项范围，跳过
        if isinstance(answer_idx, int) and answer_idx >= num_choices:
            print(f"Skipping question {i}: answer index {answer_idx} out of range (only {num_choices} choices)")
            skipped += 1
            continue

        print(f"[{idx+1}/{len(idxs)}] Processing question {i} ({num_choices} choices)...")
        
        try:
            # 安全加载图片
            image = load_image(img_data)
            if image is None:
                print(f"Skipping question {i}: could not load image")
                skipped += 1
                continue
            
            # 生成回答
            out = gen(image, question, choices, max_tokens=256, temperature=0.2)
            choice = extract_choice(out)
            
            # 如果没提取到选项，尝试其他方法
            if not choice:
                out_upper = out.upper()
                for letter in ['A', 'B', 'C', 'D'][:num_choices]:
                    if letter in out_upper:
                        choice = letter
                        break

            # 保存结果
            choice_dict = {chr(65+j): choices_list[j] for j in range(num_choices)}
            rows.append({
                'id': i,
                'question': question,
                'num_choices': num_choices,
                'choices': choice_dict,
                'model_output': out,
                'pred': choice,
                'label': label_letter
            })

            preds.append(choice)
            labels.append(label_letter)
            print(f"  → Pred: {choice}, Label: {label_letter}, {'✓' if choice == label_letter else '✗'}")
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            skipped += 1
            continue

    if len(preds) == 0:
        print("Error: No samples were successfully processed!")
        return

    acc = accuracy(preds, labels)
    print(f"\n{'='*60}")
    print(f"ScienceQA Results:")
    print(f"  Total attempted: {len(idxs)}")
    print(f"  Successfully processed: {len(preds)}")
    print(f"  Skipped/Failed: {skipped}")
    print(f"  Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print(f"  Correct: {sum(p == l for p, l in zip(preds, labels))}/{len(labels)}")
    print(f"{'='*60}\n")

    # 保存详细结果
    write_jsonl(args.out, rows)
    print(f"Detailed results saved to {args.out}")

    # 读取或创建metrics文件
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
        'total': len(labels),
        'skipped': skipped
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics updated in {metrics_path}")


if __name__ == '__main__':
    main()
