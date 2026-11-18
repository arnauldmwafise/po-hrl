"""Subgoal verification and per-segment advantages for LLM rollouts (spec Section 4.1).

Verification modes for intermediate steps (TOP is always checked against the gold answer):
  gold       ground-truth sub-answers (e.g. HiMath sub-questions)
  consensus  outcome-anchored consensus: a step's claim is verified if it matches the
             most common claim for that step among the rollouts (of the same plan) whose
             FINAL answer is correct. If no rollout is correct, the step is verified only if
             a strict majority of all rollouts agree on it (threshold tau).
  judge      a user-supplied callable judge(problem, step, context_claims, claim) -> bool,
             used for LOCAL validity (step correct given the parent claims actually used)

Credit methods (same plan, same rollouts, only the advantage differs):
  flat   outcome only, one advantage for every segment of a rollout
  chain  step credited iff it and every step before it in the WRITTEN order verified
  poset  step credited iff it and all its prerequisites verified (verified ideal)
  local  step credited iff it is valid given its parents' actual claims (judge, or re-execution:
         the frozen base model recomputes the step greedily from the same prompt; see reexec_verify)
  chain_rand  chain credit along a uniformly random linear extension of the plan, resampled for
         every (problem, training step); a control for the order the planner happened to write
For chain/poset/local, per-step credit is normalized across the G rollouts and the
normalized outcome advantage is added.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from pohrl import plan as planlib

from .checkers import equivalent, normalize
from .episode import PlannedProblem, Rollout


def _canon(c: Optional[str]) -> Optional[str]:
    return None if c is None else normalize(c)


def verify_steps(pp: PlannedProblem, rolls: Sequence[Rollout], mode: str = "consensus",
                 tau: float = 0.5, judge: Optional[Callable] = None) -> np.ndarray:
    """ok[i, j] for rollout i and step j in pp.order."""
    G, steps = len(rolls), pp.order
    ok = np.zeros((G, len(steps)), bool)
    correct = [r.correct for r in rolls]
    for j, g in enumerate(steps):
        claims = [_canon(r.segments[g].claim) if g in r.segments else None for r in rolls]
        if g == planlib.TOP:
            ok[:, j] = correct
            continue
        if mode == "gold" and pp.sub_gold and g in pp.sub_gold:
            ok[:, j] = [c is not None and equivalent(c, pp.sub_gold[g]) for c in claims]
        elif mode == "judge" and judge is not None:
            P = pp.plan.poset
            for i, r in enumerate(rolls):
                if g in r.segments:
                    ctx = {u: r.segments[u].claim for u in P.parents(g) if u in r.segments}
                    ok[i, j] = bool(judge(pp.problem, pp.plan.steps[g].text, ctx, r.segments[g].claim))
        else:  # consensus
            ref_pool = [c for c, k in zip(claims, correct) if k and c is not None]
            if ref_pool:
                ref = Counter(ref_pool).most_common(1)[0][0]
                ok[:, j] = [c is not None and equivalent(c, ref) for c in claims]
            else:
                pool = [c for c in claims if c is not None]
                if pool:
                    ref, cnt = Counter(pool).most_common(1)[0]
                    if cnt / G > tau:
                        ok[:, j] = [c is not None and equivalent(c, ref) for c in claims]
    return ok


_ORDER_RNG = random.Random(12345)


def set_order_seed(seed: int) -> None:
    _ORDER_RNG.seed(seed)


def step_credit(pp: PlannedProblem, ok: np.ndarray, method: str, rng: Optional[random.Random] = None) -> np.ndarray:
    P, order = pp.plan.poset, pp.order
    idx = {g: j for j, g in enumerate(order)}
    if method == "chain":
        return np.cumprod(ok, axis=1).astype(float)
    if method == "chain_rand":
        lin = P.sample_linear_extension(rng or _ORDER_RNG)   # one order shared by the whole group
        cols = [idx[g] for g in lin]
        c = np.zeros_like(ok, dtype=float)
        c[:, cols] = np.cumprod(ok[:, cols], axis=1)
        return c
    if method in ("poset", "local"):
        if method == "local":
            return ok.astype(float)
        c = np.zeros_like(ok, dtype=float)
        for g, j in idx.items():
            pre = [idx[u] for u in P.strict_down(g)]
            c[:, j] = ok[:, j] & (ok[:, pre].all(axis=1) if pre else True)
        return c
    raise ValueError(method)


def segment_advantages(pp: PlannedProblem, rolls: Sequence[Rollout], method: str,
                       ok: Optional[np.ndarray] = None, eps: float = 1e-6) -> List[Dict[str, float]]:
    """Advantage per (rollout, step). Returns one dict step -> advantage per rollout."""
    out_r = np.array([float(r.correct) for r in rolls])
    Ao = (out_r - out_r.mean()) / (out_r.std() + eps)
    if method == "flat":
        return [{g: float(Ao[i]) for g in r.segments} for i, r in enumerate(rolls)]
    assert ok is not None
    c = step_credit(pp, ok, method)
    Ac = (c - c.mean(0)) / (c.std(0) + eps)
    res = []
    for i, r in enumerate(rolls):
        res.append({g: float(Ac[i, j] + Ao[i]) for j, g in enumerate(pp.order) if g in r.segments})
    return res


def reexec_verify(planned: Sequence[PlannedProblem], rollouts: Sequence[Sequence[Rollout]], backend,
                  max_tokens: int = 1024, verify_mode: str = "consensus") -> List[np.ndarray]:
    """Local validity by re-execution. For every executed step, the frozen base model (no adapter)
    recomputes the step greedily from the SAME prompt the rollout's executor saw, which contains the
    parent claims that rollout actually produced; the step is valid iff its claim matches the
    recomputation. Identical prompts (e.g. root steps, or rollouts that agree on the parents) are
    recomputed once. Where the recomputation yields no claim, the consensus verdict is kept.
    The answer step is always checked against the gold answer."""
    from .checkers import extract_result
    oks = [verify_steps(pp, rolls, verify_mode) for pp, rolls in zip(planned, rollouts)]
    prompts = {}
    for pp, rolls in zip(planned, rollouts):
        for r in rolls:
            for g, seg in r.segments.items():
                if g != planlib.TOP:
                    prompts.setdefault(seg.prompt, None)
    keys = list(prompts)
    if keys:
        outs = backend.generate(keys, temperature=0.0, max_tokens=max_tokens, lora=None)
        for k, o in zip(keys, outs):
            prompts[k] = extract_result(o)
    for pp, rolls, ok in zip(planned, rollouts, oks):
        for j, g in enumerate(pp.order):
            if g == planlib.TOP:
                continue
            for i, r in enumerate(rolls):
                seg = r.segments.get(g)
                if seg is None:
                    ok[i, j] = False
                    continue
                ref = prompts.get(seg.prompt)
                if ref is not None:
                    ok[i, j] = seg.claim is not None and equivalent(seg.claim, ref)
    return oks


def training_examples(planned: Sequence[PlannedProblem], rollouts: Sequence[Sequence[Rollout]],
                      method: str, verify_mode: str = "consensus", judge=None,
                      drop_zero: bool = True, oks: Optional[List[np.ndarray]] = None) -> List[dict]:
    """Flatten to (prompt, completion, advantage, meta) examples for the trainer.
    `oks` (one verdict matrix per problem) overrides verification, e.g. from reexec_verify."""
    ex = []
    for k, (pp, rolls) in enumerate(zip(planned, rollouts)):
        if oks is not None and method != "flat":
            ok = oks[k]
        else:
            ok = None if method == "flat" else verify_steps(pp, rolls, "judge" if method == "local" else verify_mode,
                                                            judge=judge)
        advs = segment_advantages(pp, rolls, method, ok)
        P = pp.plan.poset
        for i, (r, a) in enumerate(zip(rolls, advs)):
            for g, seg in r.segments.items():
                if drop_zero and abs(a[g]) < 1e-8:
                    continue
                ex.append({"prompt": seg.prompt, "completion": seg.completion, "advantage": a[g],
                           "pid": pp.pid, "rollout": i, "step": g, "width": P.width(), "height": P.height()})
    return ex


def episode_metrics(planned: Sequence[PlannedProblem], rollouts: Sequence[Sequence[Rollout]]) -> List[dict]:
    rows = []
    for pp, rolls in zip(planned, rollouts):
        P = pp.plan.poset
        rows.append({"pid": pp.pid, "n": len(P), "width": P.width(), "height": P.height(),
                     "fallback": pp.plan.fallback, "issues": pp.plan.issues,
                     "acc": float(np.mean([r.correct for r in rolls])),
                     "rounds": float(np.mean([r.rounds for r in rolls])),
                     "words_out": float(np.mean([r.tokens_out for r in rolls])),
                     "linear_extensions": P.num_linear_extensions() if len(P) <= 16 else None})
    return rows
