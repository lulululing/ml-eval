import re
import json
from typing import List, Dict, Any

ANSWER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")
CHOICE_RE = re.compile(r"\b([ABCD])\b")


def extract_numeric_answer(text: str) -> str:
    """Extract a numeric final answer from model text; fallback to last number."""
    nums = ANSWER_RE.findall(text)
    return nums[-1] if nums else ""


def extract_choice(text: str) -> str:
    """Extract option letter A/B/C/D from model text."""
    m = CHOICE_RE.findall(text)
    return m[-1] if m else ""


def accuracy(preds: List[str], labels: List[str]) -> float:
    ok = sum(p == l for p, l in zip(preds, labels))
    return ok / max(1, len(labels))


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
