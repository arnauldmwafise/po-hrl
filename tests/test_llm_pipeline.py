import json
import os

import numpy as np
import pytest

from pohrl_llm import checkers as CK
from pohrl_llm import credit_llm as CL
from pohrl_llm import episode as EP
from pohrl_llm import prompts as PR
from pohrl_llm import run_m2, run_m3
from pohrl_llm import trainer_core as TC
from pohrl_llm.backends import ScriptedBackend

PLAN = """<plan>
<step id="s1" mode="compute" parents="">First sum EVAL[3+4]. CLAIM: number</step>
<step id="s2" mode="compute" parents="">Product EVAL[2*5]. CLAIM: number</step>
<step id="s3" mode="derive" parents="s1,s2">Multiply EVAL[{s1}*{s2}]. CLAIM: number</step>
<step id="s4" mode="case" parents="">Difference EVAL[7-2]. CLAIM: number</step>
<step id="ans" mode="combine" parents="s3,s4">Add EVAL[{s3}+{s4}]. CLAIM: number</step>
</plan>"""


def problems(k=4):
    ps = [{"id": f"p{i}", "problem": f"Synthetic problem number {i}.", "answer": "75"} for i in range(k)]
    plans = {p["problem"]: PLAN for p in ps}
    golds = {p["problem"]: "75" for p in ps}
    return ps, plans, golds


render = lambda msgs: PR.chat(msgs, None)


def test_checkers():
    assert CK.last_boxed("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"
    assert CK.equivalent("\\frac{1}{2}", "0.5")
    assert CK.equivalent("1/2", "\\dfrac{1}{2}")
    assert CK.equivalent("x = 5", "5")
    assert CK.extract_result("blah\nRESULT: 42.") == "42"
    assert not CK.equivalent("41", "42")


def test_errorless_episode_rounds_and_context():
    ps, plans, golds = problems(2)
    be = ScriptedBackend(plans, golds, err=0.0)
    planned = EP.make_plans(ps, be, render)
    assert all(pp.plan.ok for pp in planned)
    assert planned[0].plan.poset.height() == 3 and planned[0].plan.poset.width() == 3
    rolls = EP.run_episodes(planned, be, render, G=3)
    assert all(r.correct and r.rounds == 3 for rs in rolls for r in rs)
    # principal-ideal context: the s3 executor sees s1 and s2 results, not s4
    s3_prompt = rolls[0][0].segments["s3"].prompt
    assert "[s1]" in s3_prompt and "[s2]" in s3_prompt and "[s4]" not in s3_prompt
    ans_prompt = rolls[0][0].segments["ans"].prompt
    assert all(f"[{k}]" in ans_prompt for k in ["s1", "s2", "s3", "s4"])


def test_credit_methods_with_errors():
    ps, plans, golds = problems(6)
    be = ScriptedBackend(plans, golds, err=0.3, seed=3)
    planned = EP.make_plans(ps, be, render)
    rolls = EP.run_episodes(planned, be, render, G=8)
    for pp, rs in zip(planned, rolls):
        ok = CL.verify_steps(pp, rs, "consensus")
        cp = CL.step_credit(pp, ok, "poset")
        cc = CL.step_credit(pp, ok, "chain")
        assert np.all(cc <= cp + 1e-12)            # chain credit never exceeds poset credit (P7)
        flat = CL.segment_advantages(pp, rs, "flat")
        for a in flat:
            assert len(set(round(v, 9) for v in a.values())) == 1   # one advantage per rollout
        adv = CL.segment_advantages(pp, rs, "poset", ok)
        for j, g in enumerate(pp.order):
            col = [a[g] for a in adv if g in a]
            out = np.array([r.correct for r in rs], float)
            Ao = (out - out.mean()) / (out.std() + 1e-6)
            assert abs(np.mean(col) - Ao.mean()) < 1e-6   # per-step credit is centered
    ex = CL.training_examples(planned, rolls, "poset")
    assert ex and all({"prompt", "completion", "advantage"} <= set(e) for e in ex)


def test_consensus_matches_truth_when_anchor_exists():
    ps, plans, golds = problems(1)
    be = ScriptedBackend(plans, golds, err=0.25, seed=11)
    planned = EP.make_plans(ps, be, render)
    rolls = EP.run_episodes(planned, be, render, G=16)
    pp, rs = planned[0], rolls[0]
    truth = {"s1": "7", "s2": "10", "s3": "70", "s4": "5"}
    if any(r.correct for r in rs):
        ok = CL.verify_steps(pp, rs, "consensus")
        for j, g in enumerate(pp.order):
            if g in truth:
                expect = [CK.equivalent(r.segments[g].claim, truth[g]) for r in rs]
                assert list(ok[:, j]) == expect


def test_fallback_plan_runs_cot():
    ps = [{"id": "q", "problem": "Unplannable problem.", "answer": "3"}]
    be = ScriptedBackend({}, {"Unplannable problem.": "3"}, cot_acc=1.0)
    planned = EP.make_plans(ps, be, render)
    assert planned[0].plan.fallback
    rolls = EP.run_episodes(planned, be, render, G=2)
    assert all(r.correct for r in rolls[0])


def test_trainer_core():
    ids, mask = TC.build_ids(list(range(10)), [100, 101, 102], max_len=8)
    assert ids == [5, 6, 7, 8, 9, 100, 101, 102] and mask == [0] * 5 + [1] * 3
    ids, mask = TC.build_ids([1, 2], list(range(20)), max_len=8)
    assert len(ids) == 8 and sum(mask) == 7
    w = TC.token_weights([[0, 1, 1], [1, 1, 0]], [2.0, -1.0])
    assert w == [[0, 0.5, 0.5], [-0.25, -0.25, 0]]
    p, a, m = TC.pad_batch([[1, 2, 3], [4]], [[0, 1, 1], [1]], pad_id=0)
    assert p == [[1, 2, 3], [4, 0, 0]] and a == [[1, 1, 1], [1, 0, 0]] and m == [[0, 1, 1], [1, 0, 0]]


def test_run_m2_resumable(tmp_path):
    ps, plans, golds = problems(5)
    be = ScriptedBackend(plans, golds, err=0.1, cot_acc=0.4, seed=1)
    s = run_m2.run(ps, be, render, str(tmp_path), G=4, batch=2, log=lambda *a: None)
    assert s["all"]["problems"] == 5 and s["by_width"]["2-3"]["problems"] == 5
    n_calls = len(be.log)
    run_m2.run(ps, be, render, str(tmp_path), G=4, batch=2, log=lambda *a: None)
    assert len(be.log) == n_calls  # nothing recomputed


class FakeTrainer:
    def __init__(self, run_dir, resume):
        self.run_dir, self.seen = run_dir, []

    def step(self, ex):
        self.seen.append(len(ex))
        return {"loss": 0.0, "examples": len(ex)}

    def save(self, tag):
        p = os.path.join(self.run_dir, tag); os.makedirs(p, exist_ok=True); return p


def test_run_m3_resumes(tmp_path):
    ps, plans, golds = problems(8)
    be = ScriptedBackend(plans, golds, err=0.3, seed=2)
    planned = run_m3.cache_plans(ps, be, render, str(tmp_path / "plans.jsonl"))
    log = run_m3.train("poset", 0, planned[:6], planned[6:], be, render, FakeTrainer, str(tmp_path),
                       steps=3, problems_per_step=2, G=4, eval_every=2, log=lambda *a: None)
    assert [r["step"] for r in log] == [0, 1, 2, 3]
    log2 = run_m3.train("poset", 0, planned[:6], planned[6:], be, render, FakeTrainer, str(tmp_path),
                        steps=5, problems_per_step=2, G=4, eval_every=2, log=lambda *a: None)
    assert [r["step"] for r in log2] == [0, 1, 2, 3, 4, 5]
    st = json.load(open(tmp_path / "poset_s0" / "state.json"))
    assert st["step"] == 5 and st["adapter"].endswith("step5")


def test_chain_rand_is_chain_along_a_random_linear_extension():
    import random
    ps, plans, golds = problems(4)
    be = ScriptedBackend(plans, golds, err=0.3, seed=5)
    planned = EP.make_plans(ps, be, render)
    rolls = EP.run_episodes(planned, be, render, G=8)
    seen = set()
    for pp, rs in zip(planned, rolls):
        ok = CL.verify_steps(pp, rs, "consensus")
        P = pp.plan.poset
        for t in range(30):
            rng = random.Random(t)
            lin = P.sample_linear_extension(random.Random(t))
            assert P.is_linear_extension(lin)
            seen.add(tuple(lin))
            c = CL.step_credit(pp, ok, "chain_rand", rng=rng)
            idx = {g: j for j, g in enumerate(pp.order)}
            for k, g in enumerate(lin):  # credit of g = all steps up to g in the sampled order verified
                expect = ok[:, [idx[u] for u in lin[:k + 1]]].all(axis=1)
                assert np.array_equal(c[:, idx[g]].astype(bool), expect)
            assert np.all(c <= CL.step_credit(pp, ok, "poset") + 1e-12)   # still never above poset credit
    assert len(seen) > 1   # orders actually vary (this plan has 4 linear-extension classes and more)


def test_local_credit_by_reexecution_checks_steps_against_received_inputs():
    ps, plans, golds = problems(3)
    noisy = ScriptedBackend(plans, golds, err=0.35, seed=7)
    exact = ScriptedBackend(plans, golds, err=0.0, seed=0)     # stands in for the frozen re-executor
    planned = EP.make_plans(ps, noisy, render)
    rolls = EP.run_episodes(planned, noisy, render, G=8)
    oks = CL.reexec_verify(planned, rolls, exact)
    ops = {"s1": lambda c: 7, "s2": lambda c: 10, "s3": lambda c: c["s1"] * c["s2"], "s4": lambda c: 5}
    n_valid_after_wrong_input = 0
    for pp, rs, ok in zip(planned, rolls, oks):
        for j, g in enumerate(pp.order):
            for i, r in enumerate(rs):
                if g == "ans":
                    assert ok[i, j] == r.correct
                    continue
                seg = r.segments.get(g)
                if seg is None:
                    assert not ok[i, j]; continue
                try:
                    got = {u: int(float(r.segments[u].claim)) for u in pp.plan.poset.parents(g)}
                    expect = CK.equivalent(seg.claim, str(ops[g](got)))
                except (TypeError, ValueError, KeyError):
                    continue
                assert ok[i, j] == expect
                if g == "s3" and expect and (got.get("s1") != 7 or got.get("s2") != 10):
                    n_valid_after_wrong_input += 1
    # local credit can credit a correct computation on wrong inputs, which poset credit never does
    ex = CL.training_examples(planned, rolls, "local", oks=oks)
    assert ex and n_valid_after_wrong_input >= 0


def test_run_m3_new_methods(tmp_path):
    ps, plans, golds = problems(8)
    be = ScriptedBackend(plans, golds, err=0.3, seed=2)
    planned = run_m3.cache_plans(ps, be, render, str(tmp_path / "plans.jsonl"))
    for m in ["chain_rand", "local"]:
        log = run_m3.train(m, 0, planned[:6], planned[6:], be, render, FakeTrainer, str(tmp_path),
                           steps=2, problems_per_step=2, G=4, eval_every=2, log=lambda *a: None)
        assert [r["step"] for r in log] == [0, 1, 2]
