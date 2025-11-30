import argparse
import random
import json
import os
from datasets import load_dataset
from src.utils.modeling_llm import get_llm, BACKEND_HF, BACKEND_LLAMA_CPP
from src.utils.eval import extract_numeric_answer, accuracy, write_jsonl

PROMPT_TEMPLATE = (
    "You are a helpful math tutor. Solve the following grade school math problem step by step. "
    "Show your reasoning, then give the final answer as a number on the last line prefixed by 'Final answer:'.\n\n"
    "Problem: {question}\n\nReasoning:"
)


def build_prompt(q: str) -> str:
    return PROMPT_TEMPLATE.format(question=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=[BACKEND_HF, BACKEND_LLAMA_CPP], default=BACKEND_LLAMA_CPP)
    ap.add_argument('--model', type=str, help='HF model name (e.g., Qwen/Qwen2.5-7B-Instruct)')
    ap.add_argument('--gguf', type=str, help='Path to GGUF file for llama.cpp backend')
    ap.add_argument('--quantization', choices=['4bit', 'none'], default='none')
    ap.add_argument('--sample', type=int, default=50)
    ap.add_argument('--out', type=str, default='results/gsm8k.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # 确保results目录存在
    os.makedirs('results', exist_ok=True)

    random.seed(args.seed)
    print("Loading GSM8K dataset...")
    ds = load_dataset('openai/gsm8k', 'main')['test']
    idxs = random.sample(range(len(ds)), min(args.sample, len(ds)))

    print(f"Initializing LLM with backend={args.backend}...")
    gen = get_llm(args.backend, model_name=args.model, gguf_path=args.gguf, quantization=args.quantization)

    rows = []
    preds = []
    labels = []
    
    print(f"Evaluating {len(idxs)} samples...")
    for idx, i in enumerate(idxs):
        item = ds[i]
        q = item['question']
        gt = item['answer']
        prompt = build_prompt(q)
        
        print(f"[{idx+1}/{len(idxs)}] Processing question {i}...")
        try:
            out = gen(prompt, max_tokens=512, temperature=0.2)
            final = extract_numeric_answer(out)
            gt_num = extract_numeric_answer(gt)
            
            rows.append({
                'id': i, 
                'question': q, 
                'model_output': out, 
                'final_pred': final, 
                'ground_truth': gt,
                'gt_numeric': gt_num
            })
            preds.append(final)
            labels.append(gt_num)
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue

    acc = accuracy(preds, labels)
    print(f"\n{'='*50}")
    print(f"GSM8K accuracy on {len(idxs)} samples: {acc:.3f} ({acc*100:.1f}%)")
    print(f"{'='*50}\n")

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
    
    # 更新GSM8K的metrics
    metrics['gsm8k'] = {
        'samples': len(idxs), 
        'accuracy': acc,
        'correct': sum(p == l for p, l in zip(preds, labels)),
        'total': len(labels)
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics updated in {metrics_path}")


if __name__ == '__main__':
    main()
