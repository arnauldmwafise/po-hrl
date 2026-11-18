"""Framework-free pieces of the M3 policy-gradient update (tested without torch).

Loss (on-policy, one update per batch so the importance ratio is 1, as in GRPO
with a single PPO epoch):
    L = - sum_{examples e} sum_{t in completion(e)} A_e * log pi(y_t | prefix)  /  sum_e |completion(e)|
i.e. DAPO-style token-level normalization over the whole batch.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def build_ids(prompt_ids: Sequence[int], completion_ids: Sequence[int], max_len: int) -> Tuple[List[int], List[int]]:
    """Concatenate and build a loss mask that is 1 only on completion tokens.
    Truncates from the LEFT of the prompt if too long, never the completion."""
    comp = list(completion_ids)[: max_len - 1]
    room = max_len - len(comp)
    prm = list(prompt_ids)[-room:] if room > 0 else []
    ids = prm + comp
    mask = [0] * len(prm) + [1] * len(comp)
    return ids, mask


def pad_batch(seqs: Sequence[Sequence[int]], masks: Sequence[Sequence[int]], pad_id: int):
    L = max(len(s) for s in seqs)
    ids = [list(s) + [pad_id] * (L - len(s)) for s in seqs]
    attn = [[1] * len(s) + [0] * (L - len(s)) for s in seqs]
    lm = [list(m) + [0] * (L - len(m)) for m in masks]
    return ids, attn, lm


def token_weights(masks: Sequence[Sequence[int]], advantages: Sequence[float]) -> List[List[float]]:
    """Per-token weights A_e / total_completion_tokens, aligned with masks (shifted by the caller)."""
    total = sum(sum(m) for m in masks)
    if total == 0:
        return [[0.0] * len(m) for m in masks]
    return [[a * x / total for x in m] for m, a in zip(masks, advantages)]


def micro_batches(n: int, size: int):
    for i in range(0, n, size):
        yield list(range(i, min(n, i + size)))
