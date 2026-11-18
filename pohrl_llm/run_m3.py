"""M3: executor RL with poset vs chain vs flat credit; the planner is frozen.

Design for a fair comparison:
  * plans are generated ONCE by the frozen base model and cached (m3_plans.jsonl);
    every method trains on exactly the same plans and problems, in the same order;
  * methods differ only in the per-segment advantage (credit_llm);
  * evaluation: greedy-ish (temperature 0.6, G_eval samples) PO-HRL accuracy on a held-out
    set every `eval_every` steps, split by plan-width bucket (P6 predicts the gap grows with width).
Checkpoints (adapter + optimizer-free state + step + log) go to out_dir/<method>_s<seed>/,
so a disconnected Colab session resumes where it stopped.
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable, List

import numpy as np

from pohrl import plan as planlib

from . import credit_llm as CL
from . import episode as EP
from .run_m2 import width_bucket


def cache_plans(problems: List[dict], backend, render, path: str, batch: int = 32, plan_thinking: bool = True,
                plan_max_tokens: int = 4096) -> List[EP.PlannedProblem]:
    """Plan every problem once with the frozen model and cache the raw planner output.
    Same planner settings as M2 run 2: thinking mode + budget forcing, no dependency check."""
    from . import prompts as PR
    tok = getattr(backend, "tokenizer", None)
    plan_render = (lambda m: PR.chat(m, tok, enable_thinking=True)) if (plan_thinking and tok is not None) else None
    have = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l); have[r["pid"]] = r
    todo = [p for p in problems if p["id"] not in have]
    for s in range(0, len(todo), batch):
        for pp in EP.make_plans(todo[s:s + batch], backend, render, plan_render=plan_render,
                                max_tokens=plan_max_tokens if plan_render else 768, audit=False):
            r = {"pid": pp.pid, "plan": pp.planner_completion}
            have[pp.pid] = r
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")
    out = []
    for p in problems:
        v = planlib.validate(have[p["id"]]["plan"])
        out.append(EP.PlannedProblem(p["id"], p["problem"], p["answer"], v, EP.written_order(v),
                                     "", have[p["id"]]["plan"], p.get("sub_answers")))
    return out


def stratify(planned: List[EP.PlannedProblem], per_bucket: int, seed: int = 0) -> List[EP.PlannedProblem]:
    """Evaluation set with up to `per_bucket` problems from each width bucket (1, 2-3, 4+); fallbacks excluded.
    Chains (width 1) are the control; wider plans are where Theorem 1 predicts a gap."""
    rng = random.Random(seed)
    by = {}
    for pp in planned:
        if not pp.plan.fallback:
            by.setdefault(width_bucket(pp.plan.poset.width()), []).append(pp)
    out = []
    for b in ["1", "2-3", "4+"]:
        xs = by.get(b, [])
        rng.shuffle(xs)
        out += xs[:per_bucket]
    return out


def planned_only(planned: List[EP.PlannedProblem]) -> List[EP.PlannedProblem]:
    return [pp for pp in planned if not pp.plan.fallback]


def evaluate(planned, backend, render, lora, G_eval=4, temperature=0.6, max_tokens=1024):
    rolls = EP.run_episodes(planned, backend, render, G_eval, temperature=temperature, max_tokens=max_tokens, lora=lora)
    rows = CL.episode_metrics(planned, rolls)
    out = {"acc": float(np.mean([r["acc"] for r in rows]))}
    for b in ["1", "2-3", "4+"]:
        sel = [r["acc"] for r in rows if not r["fallback"] and width_bucket(r["width"]) == b]
        out[f"acc_w{b}"] = float(np.mean(sel)) if sel else None
        out[f"n_w{b}"] = len(sel)
    return out


def train(method: str, seed: int, train_planned, eval_planned, backend, render: Callable, trainer_factory,
          out_dir: str, steps: int = 200, problems_per_step: int = 16, G: int = 8, eval_every: int = 25,
          verify_mode: str = "consensus", judge=None, temperature: float = 1.0, max_tokens: int = 1024, log=print):
    run_dir = os.path.join(out_dir, f"{method}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    state_path = os.path.join(run_dir, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"step": 0, "adapter": None, "log": []}
    trainer = trainer_factory(run_dir, state["adapter"])
    lora_id = 1 + 1_000_000 * seed + 100_000 * {"flat": 0, "chain": 1, "poset": 2, "local": 3, "chain_rand": 4}[method]
    lora = None if state["adapter"] is None else (f"{method}{seed}", lora_id + state["step"], state["adapter"])
    rng = random.Random(seed)
    order = list(range(len(train_planned)))
    rng.shuffle(order)
    if state["step"] == 0:
        ev = evaluate(eval_planned, backend, render, lora); ev["step"] = 0
        state["log"].append(ev); log(f"{method} s{seed} step 0 {ev}")
    for step in range(state["step"] + 1, steps + 1):
        idx = [order[(step * problems_per_step + k) % len(order)] for k in range(problems_per_step)]
        batch = [train_planned[i] for i in idx]
        rolls = EP.run_episodes(batch, backend, render, G, temperature=temperature, max_tokens=max_tokens, lora=lora)
        if method == "chain_rand":
            CL.set_order_seed(1_000_003 * seed + step)  # resumable: the orders depend only on (seed, step)
        oks = CL.reexec_verify(batch, rolls, backend, max_tokens=max_tokens, verify_mode=verify_mode) \
            if (method == "local" and judge is None) else None
        ex = CL.training_examples(batch, rolls, method, verify_mode=verify_mode, judge=judge, oks=oks)
        stats = trainer.step(ex)
        path = trainer.save(f"step{step}")
        lora = (f"{method}{seed}", lora_id + step, path)
        rec = {"step": step, "train_acc": float(np.mean([r.correct for rs in rolls for r in rs])), **stats}
        if step % eval_every == 0 or step == steps:
            rec.update(evaluate(eval_planned, backend, render, lora))
            log(f"{method} s{seed} {rec}")
        state.update({"step": step, "adapter": path}); state["log"].append(rec)
        json.dump(state, open(state_path, "w"))
        prev = os.path.join(run_dir, f"step{step - 1}")
        if step > 1 and os.path.isdir(prev) and (step - 1) % eval_every != 0:
            import shutil; shutil.rmtree(prev, ignore_errors=True)  # keep only eval checkpoints
    return state["log"]
