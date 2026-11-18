"""PO-HRL episodes with a real (or mock) language model.

Backend interface (implemented by VLLMBackend for Colab and ScriptedBackend for tests):
    backend.generate(prompts: list[str], temperature: float, max_tokens: int,
                     lora: object | None = None) -> list[str]
One completion per prompt; repeat a prompt to sample it several times.

An episode follows spec Section 3: plan (or a fixed plan), then rounds over the
frontier. All frontier steps of all rollouts of all problems in a batch are sent
to the backend together, so a round is one batched generate call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pohrl import plan as planlib
from pohrl.poset import Poset

from . import prompts as PR
from .checkers import equivalent, extract_result, last_boxed


@dataclass
class Segment:
    step: str
    prompt: str
    completion: str
    claim: Optional[str]
    round: int


@dataclass
class Rollout:
    segments: Dict[str, Segment] = field(default_factory=dict)
    answer: Optional[str] = None
    correct: bool = False
    rounds: int = 0
    tokens_out: int = 0
    accepted: List[str] = field(default_factory=list)


@dataclass
class PlannedProblem:
    pid: str
    problem: str
    gold: str
    plan: planlib.ValidationResult
    order: List[str]                 # the planner's written order (a linear extension)
    planner_prompt: str = ""
    planner_completion: str = ""
    sub_gold: Optional[Dict[str, str]] = None  # ground-truth sub-answers (e.g. HiMath), optional


def written_order(v: planlib.ValidationResult) -> List[str]:
    """Order in which the planner wrote the steps, repaired into a linear extension."""
    P = v.poset
    pending = list(v.steps.keys())
    out: List[str] = []
    while pending:
        for g in pending:
            if P.strict_down(g) <= set(out):
                out.append(g); pending.remove(g); break
        else:  # cannot happen for a valid poset
            raise RuntimeError("no linear extension")
    return out


def _has_plan(text: str) -> bool:
    i = text.rfind("<plan>")
    return i >= 0 and "</plan>" in text[i:]


def make_plans(problems: Sequence[dict], backend, render: Callable, n_max=12, temperature=0.0,
               max_tokens=768, plan_render: Callable = None, force_max_tokens: int = 1024,
               max_think_chars: int = 12000, audit: bool = False, audit_max_tokens: int = 1024
               ) -> List[PlannedProblem]:
    """Plan every problem.
    plan_render: optional renderer for the planner call only (e.g. Qwen3 thinking on).
    Budget forcing: if an output contains no complete <plan> block (typically the model kept
    thinking until max_tokens), its thinking is cut, closed with </think>, and the model is made
    to continue from '<plan>' with a short non-thinking budget.
    audit: one extra call per problem asks the model to correct the parents of its own plan;
    the audited plan is used if it validates with the same step ids, and both are recorded."""
    pr_fn = plan_render or render
    prompts = [pr_fn(PR.planner_messages(p["problem"], n_max)) for p in problems]
    outs = backend.generate(prompts, temperature=temperature, max_tokens=max_tokens)
    forced = [False] * len(outs)
    redo = [i for i, o in enumerate(outs) if not _has_plan(o)]
    if redo:
        cont = []
        for i in redo:
            o = outs[i]
            think = o.split("</think>")[0] if "</think>" in o else o
            think = think[:max_think_chars]
            cont.append(prompts[i] + think + "\n</think>\n\n<plan>\n")
        gen = backend.generate(cont, temperature=0.0, max_tokens=force_max_tokens)
        for i, g in zip(redo, gen):
            outs[i] = outs[i][:max_think_chars] + "\n</think>\n\n<plan>\n" + g
            forced[i] = True
    res = []
    for k, (p, pr, o) in enumerate(zip(problems, prompts, outs)):
        v = planlib.validate(o)
        pp = PlannedProblem(p["id"], p["problem"], p["answer"], v, written_order(v), pr, o, p.get("sub_answers"))
        pp.forced = forced[k]
        pp.pre_audit = None
        res.append(pp)
    if audit:
        idx = [k for k, pp in enumerate(res) if pp.plan.ok]
        if idx:
            aprompts = [render(PR.audit_messages(res[k].problem, planlib.render_plan(list(res[k].plan.steps.values()))))
                        for k in idx]
            aouts = backend.generate(aprompts, temperature=0.0, max_tokens=audit_max_tokens)
            for k, a in zip(idx, aouts):
                va = planlib.validate(a)
                pp = res[k]
                if va.ok and set(va.steps) == set(pp.plan.steps):
                    pp.pre_audit = pp.plan
                    pp.plan = va
                    pp.order = written_order(va)
                    pp.planner_completion = pp.planner_completion + "\n<!-- audit -->\n" + a
    return res


def run_episodes(planned: Sequence[PlannedProblem], backend, render: Callable, G: int,
                 temperature=1.0, max_tokens=1024, max_rounds=12, lora=None) -> List[List[Rollout]]:
    """Execute every planned problem G times. No replanning (planner frozen, as in M2/M3)."""
    rolls = [[Rollout() for _ in range(G)] for _ in planned]
    done = [[set() for _ in range(G)] for _ in planned]
    for rnd in range(1, max_rounds + 1):
        jobs = []
        for pi, pp in enumerate(planned):
            P = pp.plan.poset
            for gi in range(G):
                if planlib.TOP in done[pi][gi]:
                    continue
                for g in P.frontier(done[pi][gi]):
                    st = pp.plan.steps[g]
                    ctx = {u: (rolls[pi][gi].segments[u].claim or "(no result)")
                           for u in pp.order if u in P.strict_down(g)}
                    cdesc = {u: pp.plan.steps[u].text for u in ctx}
                    if pp.plan.fallback:
                        msgs = PR.cot_messages(pp.problem)
                    else:
                        msgs = PR.executor_messages(pp.problem, st.text, st.mode, ctx, cdesc, g == planlib.TOP)
                    jobs.append((pi, gi, g, render(msgs)))
        if not jobs:
            break
        outs = backend.generate([j[3] for j in jobs], temperature=temperature, max_tokens=max_tokens, lora=lora)
        for (pi, gi, g, pr), o in zip(jobs, outs):
            claim = extract_result(o) if not planned[pi].plan.fallback else last_boxed(o)
            r = rolls[pi][gi]
            r.segments[g] = Segment(g, pr, o, claim, rnd)
            r.tokens_out += len(o.split())
            done[pi][gi].add(g)  # accept every executed step (verification is done afterwards)
            r.rounds = rnd
    for pi, pp in enumerate(planned):
        for r in rolls[pi]:
            top = r.segments.get(planlib.TOP)
            r.answer = top.claim if top else None
            r.correct = equivalent(r.answer, pp.gold)
            r.accepted = list(r.segments)
    return rolls


def run_cot(problems: Sequence[dict], backend, render: Callable, G: int, temperature=1.0,
            max_tokens=2048, lora=None) -> List[List[Rollout]]:
    prompts = [render(PR.cot_messages(p["problem"])) for p in problems for _ in range(G)]
    outs = backend.generate(prompts, temperature=temperature, max_tokens=max_tokens, lora=lora)
    res = []
    for i, p in enumerate(problems):
        rs = []
        for o, pr in zip(outs[i * G:(i + 1) * G], prompts[i * G:(i + 1) * G]):
            ans = last_boxed(o)
            r = Rollout({planlib.TOP: Segment(planlib.TOP, pr, o, ans, 1)}, ans, equivalent(ans, p["answer"]), 1,
                        len(o.split()), [planlib.TOP])
            rs.append(r)
        res.append(rs)
    return res
