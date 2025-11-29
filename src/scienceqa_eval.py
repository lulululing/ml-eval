import argparse
import random
import json
from typing import Tuple

from datasets import load_dataset
from PIL import Image

from src.utils.modeling_mllm import get_mllm, BACKEND_HF, BACKEND_LLAVA_CPP
from src.utils.eval import extract_choice, accuracy, write_jsonl

PROMPT_SUFFIX = "Please choose the correct option (A/B/C/D)."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=[BACKEND_HF, BACKEND_LLAVA_CPP], default=BACKEND_HF)
    ap.add_argument('--model', type=str, help='HF MLLM model name (e.g., Qwen/Qwen2-VL-7B-Instruct)')
    ap.add_argument('--gguf', type=str, help='Path to LLaVA GGUF if using llava.cpp backend')
    ap.add_argument('--sample', type=int, default=50)
    ap.add_argument('--out', type=str, default='results/scienceqa.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    ds = load_dataset('derek-thomas/ScienceQA', 'scienceqa')['test']
    # 仅包含image字段不为空的样本
    image_idxs = [i for i in range(len(ds)) if ds[i].get('image')]
    idxs = random.sample(image_idxs, args.sample)

    gen = get_mllm(args.backend, model_name=args.model, gguf_path=args.gguf)

    rows = []
    preds = []
    labels = []

    for i in idxs:
        item = ds[i]
        img_path = item['image']
        question = item['question']
        choices = (item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3])
        label_letter = item['answer']  # already A/B/C/D in this dataset split

        image = Image.open(img_path).convert('RGB')
        text = question + "\n" + PROMPT_SUFFIX
        out = gen(image, question, choices, max_tokens=256, temperature=0.2)
        choice = extract_choice(out)

        rows.append({
            'id': i,
            'image': img_path,
            'question': question,
            'choices': {'A': choices[0], 'B': choices[1], 'C': choices[2], 'D': choices[3]},
            'model_output': out,
            'pred': choice,
            'label': label_letter
        })

        preds.append(choice)
        labels.append(label_letter)

    acc = accuracy(preds, labels)
    print(f"ScienceQA (image-only) accuracy on {len(idxs)} samples: {acc:.3f}")

    write_jsonl(args.out, rows)
    with open('results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'scienceqa': {'samples': len(idxs), 'accuracy': acc}}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
