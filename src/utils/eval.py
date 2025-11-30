import re
import json
from typing import List, Dict, Any

ANSWER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")
CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def extract_numeric_answer(text: str) -> str:
    """
    从模型输出中提取数字答案
    优先查找"Final answer:"后的数字，否则返回最后一个数字
    """
    if not text:
        return ""

    # 尝试查找 "Final answer:" 后面的数字
    final_answer_pattern = r"[Ff]inal\s+[Aa]nswer\s*[:：]\s*(\d+(?:\.\d+)?)"
    match = re.search(final_answer_pattern, text)
    if match:
        return match.group(1)

    # 尝试查找 "答案是" 或 "答案为" 后面的数字（支持中文）
    chinese_answer_pattern = r"答案[是为]\s*(\d+(?:\.\d+)?)"
    match = re.search(chinese_answer_pattern, text)
    if match:
        return match.group(1)

    # 否则返回最后一个数字
    nums = ANSWER_RE.findall(text)
    return nums[-1] if nums else ""


def extract_choice(text: str) -> str:
    """
    从模型输出中提取选项字母 A/B/C/D
    优先查找明确的答案标记，否则返回最后一个出现的选项
    """
    if not text:
        return ""

    text_upper = text.upper()

    # 尝试查找明确的答案模式
    patterns = [
        r"ANSWER\s*[:：是为]\s*([ABCD])",
        r"答案\s*[:：是为]\s*([ABCD])",
        r"选择\s*[:：]\s*([ABCD])",
        r"CORRECT\s+(?:ANSWER|OPTION)\s*[:：是为]?\s*([ABCD])",
        r"^([ABCD])(?:\.|:|：)",  # 开头就是选项
    ]

    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)

    # 否则返回最后一个出现的选项字母
    matches = CHOICE_RE.findall(text_upper)
    return matches[-1] if matches else ""


def accuracy(preds: List[str], labels: List[str]) -> float:
    """计算准确率"""
    if not labels:
        return 0.0
    correct = sum(p.strip().upper() == l.strip().upper() for p, l in zip(preds, labels) if p and l)
    return correct / len(labels)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    """写入JSONL文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取JSONL文件"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
