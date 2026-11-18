"""M2: inference-only PO-HRL vs chain-of-thought, with a frozen model.

Outputs (resumable; one JSON line per problem) in out_dir:
  m2_plans.jsonl     planner output, validation issues, width/height/e(P)
  m2_results.jsonl   per problem: PO-HRL accuracy/rounds/words and CoT accuracy/words
  m2_summary.json    aggregate table, also split by plan width bucket
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable, List

import numpy as np

from . import credit_llm as CL
from . import episode as EP


def _done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(l)["pid"] for l in f if l.strip()}


def width_bucket(w: int) -> str:
    return "1" if w <= 1 else ("2-3" if w <= 3 else "4+")


def run(problems: List[dict], backend, render: Callable, out_dir: str, G: int = 4, batch: int = 16,
        temperature: float = 0.7, plan_temperature: float = 0.0, max_tokens: int = 1024,
        cot_max_tokens: int = 2048, plan_thinking: bool = True, plan_max_tokens: int = 4096,
        audit: bool = True, log=print) -> dict:
    """plan_thinking: render the planner prompt with Qwen3 thinking enabled (plans are made once
    per problem, so this is cheap); executors and CoT stay in non-thinking mode."""
    from . import prompts as PR
    tok = getattr(backend, "tokenizer", None)
    plan_render = (lambda m: PR.chat(m, tok, enable_thinking=True)) if (plan_thinking and tok is not None) else None
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "m2_results.jsonl")
    plan_path = os.path.join(out_dir, "m2_plans.jsonl")
    done = _done_ids(res_path)
    todo = [p for p in problems if p["id"] not in done]
    log(f"M2: {len(done)} done, {len(todo)} to go")
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        planned = EP.make_plans(chunk, backend, render, temperature=plan_temperature,
                                max_tokens=plan_max_tokens if plan_render else 768, plan_render=plan_render,
                                audit=audit)
        rolls = EP.run_episodes(planned, backend, render, G, temperature=temperature, max_tokens=max_tokens)
        cot = EP.run_cot(chunk, backend, render, G, temperature=temperature, max_tokens=cot_max_tokens)
        mets = CL.episode_metrics(planned, rolls)
        with open(plan_path, "a") as fp, open(res_path, "a") as fr:
            for pp, m, c in zip(planned, mets, cot):
                raw = __import__("pohrl").plan.validate(pp.planner_completion, orphan_policy="drop")
                fp.write(json.dumps({"pid": pp.pid, "plan": pp.planner_completion[-4000:], "issues": pp.plan.issues,
                                     "raw_n_kept": len(raw.poset), "raw_width": raw.poset.width(),
                                     "n_steps": len(pp.plan.poset),
                                     "forced": getattr(pp, "forced", False),
                                     "audited": getattr(pp, "pre_audit", None) is not None,
                                     "pre_audit_width": (pp.pre_audit.poset.width() if getattr(pp, "pre_audit", None) else None),
                                     "pre_audit_edges": ({g: sorted(pp.pre_audit.poset.parents(g)) for g in pp.pre_audit.poset.elements}
                                                         if getattr(pp, "pre_audit", None) else None),
                                     "fallback": pp.plan.fallback, "order": pp.order,
                                     "edges": {g: sorted(pp.plan.poset.parents(g)) for g in pp.order}}) + "\n")
                fr.write(json.dumps({**m, "cot_acc": float(np.mean([r.correct for r in c])),
                                     "cot_words": float(np.mean([r.tokens_out for r in c]))}) + "\n")
        log(f"M2: {min(s + batch, len(todo))}/{len(todo)}")
    return summarize(out_dir)


def summarize(out_dir: str) -> dict:
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "m2_results.jsonl")) if l.strip()]
    def agg(rs):
        return {"problems": len(rs), "pohrl_acc": float(np.mean([r["acc"] for r in rs])),
                "cot_acc": float(np.mean([r["cot_acc"] for r in rs])),
                "mean_rounds": float(np.mean([r["rounds"] for r in rs])),
                "pohrl_words": float(np.mean([r["words_out"] for r in rs])),
                "cot_words": float(np.mean([r["cot_words"] for r in rs])),
                "fallback_rate": float(np.mean([r["fallback"] for r in rs])),
                "median_width": float(np.median([r["width"] for r in rs])),
                "share_width_ge2": float(np.mean([r["width"] >= 2 for r in rs]))}
    by = defaultdict(list)
    for r in rows:
        if not r["fallback"]:
            by[width_bucket(r["width"])].append(r)
    summ = {"all": agg(rows), "by_width": {k: agg(v) for k, v in sorted(by.items())}}
    with open(os.path.join(out_dir, "m2_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    return summ
