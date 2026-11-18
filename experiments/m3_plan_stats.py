"""Incomparable-pair statistics of the cached M3 plans (Section 7 and Proposition 4 of the paper).

For every plan written by Qwen3-1.7B this computes iota(P), the number of incomparable pairs, and the
share of steps with at least one incomparable predecessor under the written order and under uniformly
random linear extensions. It also checks the identity sum_g |I_L(g)| = iota(P) for every order used.

Usage:
    python experiments/m3_plan_stats.py [results/m3 directory] [--eval-pids reeval.jsonl]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pohrl import plan as planlib  # noqa: E402
from pohrl_llm.episode import written_order  # noqa: E402


def load_plans(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return {json.loads(l)["pid"]: json.loads(l)["plan"] for l in f if l.strip()}


def exposure(P, order):
    pos = {g: i for i, g in enumerate(order)}
    counts = []
    for g in order:
        sd = P.strict_down(g)
        counts.append(sum(1 for u in order[:pos[g]] if u not in sd))
    return counts


def iota(P):
    els = list(P.elements)
    return sum(1 for i, a in enumerate(els) for b in els[i + 1:] if a not in P.down(b) and b not in P.down(a))


def stats(plan_texts, n_random=20, seed=0):
    rng = random.Random(seed)
    steps = pairs = exp_written = exp_random = n_rand_steps = 0
    for text in plan_texts:
        v = planlib.validate(text)
        if v.fallback:
            continue
        P = v.poset
        k = iota(P)
        n = len(list(P.elements))
        steps += n
        pairs += k
        cw = exposure(P, written_order(v))
        assert sum(cw) == k, "identity violated (written order)"
        exp_written += sum(1 for c in cw if c > 0)
        for _ in range(n_random):
            cr = exposure(P, P.sample_linear_extension(rng))
            assert sum(cr) == k, "identity violated (random order)"
            exp_random += sum(1 for c in cr if c > 0)
            n_rand_steps += n
    return {"steps": steps, "pairs_per_step": pairs / steps, "share_exposed_written": exp_written / steps,
            "share_exposed_random": exp_random / n_rand_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(ROOT / "experiments" / "results" / "m3"))
    args = ap.parse_args()
    res = Path(args.results)
    train = load_plans(next(res.glob("plans_train.jsonl*")))
    ev = load_plans(next(res.glob("plans_eval.jsonl*")))
    eval_pids = [r["pid"] for r in json.loads((res / "reeval.jsonl").open().readline())["rows"]]
    out = {"train": stats(train.values()), "eval_117": stats([ev[p] for p in eval_pids])}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
