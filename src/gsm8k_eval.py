import argparse
import random
import json
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

    random.seed(args.seed)
    ds = load_dataset('openai/gsm8k', 'main')['test']
    idxs = random.sample(range(len(ds)), args.sample)

    gen = get_llm(args.backend, model_name=args.model, gguf_path=args.gguf, quantization=args.quantization)

    rows = []
    preds = []
    labels = []
    for i in idxs:
        item = ds[i]
        q = item['question']
        gt = item['answer']
        prompt = build_prompt(q)
        out = gen(prompt, max_tokens=512, temperature=0.2)
        final = extract_numeric_answer(out) or extract_numeric_answer(gt)
        rows.append({'id': i, 'question': q, 'model_output': out, 'final_pred': final, 'ground_truth': gt})
        preds.append(final)
        labels.append(extract_numeric_answer(gt))

    acc = accuracy(preds, labels)
    print(f"GSM8K accuracy on {len(idxs)} samples: {acc:.3f}")

    write_jsonl(args.out, rows)
    with open('results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'gsm8k': {'samples': len(idxs), 'accuracy': acc}}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
