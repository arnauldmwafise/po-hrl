"""Dataset loading. Hugging Face dataset ids are configurable because hub names change;
the defaults below should be checked on the hub before a run. Any JSONL file with
fields {id, problem, answer[, sub_answers]} also works."""
from __future__ import annotations

import json
from typing import List, Optional

from .checkers import last_boxed

DEFAULTS = {
    "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer"),
    "aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
    "gsm8k": ("openai/gsm8k", "test", "question", "answer"),
}


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_hf(name: str, split: Optional[str] = None, qkey: Optional[str] = None, akey: Optional[str] = None,
            config: Optional[str] = None, limit: Optional[int] = None, filter_fn=None) -> List[dict]:
    from datasets import load_dataset
    if name in DEFAULTS:
        name, split0, q0, a0 = DEFAULTS[name]
        split, qkey, akey = split or split0, qkey or q0, akey or a0
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    out = []
    for i, r in enumerate(ds):
        ans = str(r[akey])
        if "####" in ans:                     # GSM8K format
            ans = ans.split("####")[-1].strip()
        elif "\\boxed" in ans:                # MATH solutions
            ans = last_boxed(ans) or ans
        row = {"id": str(r.get("unique_id", r.get("id", i))), "problem": r[qkey], "answer": ans}
        if "level" in r:
            row["level"] = r["level"]
        if filter_fn is None or filter_fn(row):
            out.append(row)
        if limit and len(out) >= limit:
            break
    return out
