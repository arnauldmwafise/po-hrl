"""Milestone M1: poset statistics of DAG-MATH gold solutions.

Input: the DAG-MATH-Formatted-CoT JSON files (Hugging Face dataset
liminho123/DAG-MATH-Formatted-CoT). Schema (from the DAG-MATH README): each file is
a list with one object {problem_id, domain, difficulty (1-6), problem_text,
sample_id, final_answer, steps:[{step_id, edge, direct_dependent_steps, node}]}.

Usage:
    python experiments/dagmath_stats.py <dir-or-files...> [--out stats.csv]

For each solution: n, height, width, #ideals, #linear extensions (log10),
prefix-tree size, context ratio (principal ideal vs full history), and the mean
chain-credit attenuation of Proposition P6 at p = 0.75 under the solution's own
written order. Summary by difficulty is printed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pohrl import Poset  # noqa: E402

MAX_IDEALS = 200_000  # skip exact lattice counts beyond this (report None)


def load(paths):
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
        for f in files:
            data = json.loads(f.read_text())
            for obj in (data if isinstance(data, list) else [data]):
                if isinstance(obj, dict) and "steps" in obj:
                    yield obj


def to_poset(obj):
    steps = obj["steps"]
    ids = [str(s["step_id"]) for s in steps]
    idset = set(ids)
    parents = {str(s["step_id"]): [str(d) for d in (s.get("direct_dependent_steps") or []) if str(d) in idset]
               for s in steps}
    P = Poset(ids, parents)
    sinks = P.maximal()
    if len(sinks) > 1:  # add a TOP that combines all final conclusions
        P = Poset(ids + ["TOP"], {**parents, "TOP": sinks})
        ids = ids + ["TOP"]
    return P, ids


def stats(P: Poset, written_order, p=0.75):
    out = {"n": len(P), "height": P.height(), "width": P.width()}
    ideals = P.ideals() if len(P) <= 40 else None
    if ideals is not None and len(ideals) <= MAX_IDEALS:
        out["ideals"] = len(ideals)
        e = P.num_linear_extensions()
        out["log10_linear_extensions"] = round(math.log10(e), 3)
        out["prefix_tree_nodes"] = P.prefix_tree_size()
    else:
        out["ideals"] = out["log10_linear_extensions"] = out["prefix_tree_nodes"] = None
    pos = {g: i for i, g in enumerate(written_order)}
    ctx_ideal = sum(len(P.strict_down(g)) for g in P.elements)
    ctx_seq = sum(pos[g] for g in P.elements)
    out["context_ratio"] = round(ctx_ideal / ctx_seq, 4) if ctx_seq else 1.0
    att = []
    for g in P.elements:
        k = sum(1 for u in written_order[: pos[g]] if not P.comparable(u, g))
        att.append(p ** k)
    out["mean_attenuation_p075"] = round(sum(att) / len(att), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="dagmath_poset_stats.csv")
    a = ap.parse_args()
    rows, bad = [], 0
    for obj in load(a.paths):
        try:
            P, order = to_poset(obj)
        except ValueError:
            bad += 1
            continue
        r = {"problem_id": obj.get("problem_id"), "sample_id": obj.get("sample_id"),
             "difficulty": obj.get("difficulty"), **stats(P, order)}
        rows.append(r)
    if not rows:
        print("no solutions found"); return
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["difficulty"]].append(r)
    print(f"{len(rows)} solutions ({bad} skipped: cycles or bad ids) -> {a.out}")
    print("difficulty  count  median_n  median_height  median_width  share_width>=2  mean_attenuation")
    for d in sorted(by, key=lambda x: (x is None, x)):
        rs = by[d]
        med = lambda k: sorted(x[k] for x in rs)[len(rs) // 2]
        share = sum(x["width"] >= 2 for x in rs) / len(rs)
        att = sum(x["mean_attenuation_p075"] for x in rs) / len(rs)
        print(f"{str(d):>10}  {len(rs):5d}  {med('n'):8}  {med('height'):13}  {med('width'):12}  {share:14.2f}  {att:16.3f}")


if __name__ == "__main__":
    main()
