import math
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pohrl import plan as planlib
from pohrl_llm import wide, episode as EP, credit_llm as CL
from pohrl_llm.checkers import equivalent


def iota(P):
    els = list(P.elements)
    return sum(1 for i, a in enumerate(els) for b in els[i + 1:] if a not in P.down(b) and b not in P.down(a))


def test_gold_plans_have_the_intended_shape():
    for k in range(1, 7):
        for p in wide.make_set(20, [k], seed=k, prefix=f"k{k}_"):
            v = planlib.validate(p["plan"])
            assert not v.fallback and not v.issues, v.issues
            P = v.poset
            assert len(list(P.elements)) == 2 * k + 1
            assert P.width() == k and P.height() == 3
            assert iota(P) == 4 * math.comb(k, 2)
            assert abs(iota(P) / (2 * k + 1) - wide.incomparable_pairs_per_step(k)) < 1e-12
            order = EP.written_order(v)
            assert P.is_linear_extension(order)
            assert set(p["sub_answers"]) == set(P.elements)


def test_answers_are_consistent():
    for p in wide.make_set(300, [1, 2, 3, 4, 5, 6], seed=0, prefix="c"):
        ys = [int(p["sub_answers"][f"{c.lower()}2"]) for c in wide.PARTS[:p["k"]]]
        assert int(p["answer"]) == sum(ys) == int(p["sub_answers"]["ans"])
        assert f"Compute $" in p["problem"]


def test_templates_ground_truth_by_brute_force():
    r = random.Random(3)
    for t in wide.TEMPLATES:
        for _ in range(30):
            text, d1, d2, x, y = t(r, "A")
            assert isinstance(x, int) and isinstance(y, int)
            assert "x_A" in text and "y_A" in text


def test_gold_verification_and_credit_on_a_wide_plan():
    p = wide.make_problem(4, random.Random(1), "t0")
    v = planlib.validate(p["plan"])
    pp = EP.PlannedProblem(p["id"], p["problem"], p["answer"], v, EP.written_order(v), "", p["plan"], p["sub_answers"])

    class Seg:
        def __init__(self, claim): self.claim = claim

    class Roll:
        def __init__(self, segs, correct): self.segments, self.correct = segs, correct

    good = {g: Seg(p["sub_answers"][g]) for g in pp.order}
    bad_a = dict(good); bad_a["a1"] = Seg("123456789"); bad_a["a2"] = Seg("-1"); bad_a["ans"] = Seg("0")
    rolls = [Roll(good, True), Roll(bad_a, False), Roll({**good, "b2": Seg(None)}, False)]
    ok = CL.verify_steps(pp, rolls, "gold")
    idx = {g: j for j, g in enumerate(pp.order)}
    assert ok[0].all()
    assert not ok[1, idx["a1"]] and not ok[1, idx["a2"]] and ok[1, idx["b1"]] and ok[1, idx["d2"]]
    assert not ok[2, idx["b2"]]
    poset = CL.step_credit(pp, ok, "poset")
    # in rollout 1 part A failed: poset credit keeps parts B, C, D, chain credit along an order placing A first does not
    assert poset[1, idx["b2"]] == 1 and poset[1, idx["a2"]] == 0


def test_checker_takes_value_after_last_equals():
    from pohrl_llm.checkers import equivalent, extract_result
    assert equivalent("x_A = 245", "245")
    assert equivalent("y_A + y_B + y_C = 3 + 12 + 5 = 20", "20")
    assert equivalent(extract_result("so\nRESULT: y_A + y_B = 3 + 17 = 20"), "20")
    assert equivalent("$y_A = 7$", "7") and equivalent("x=5", "5") and equivalent("y=2x+3", "y=2x+3")
    assert not equivalent("x_A = 244", "245")


def test_stored_multipart_data_match_generator():
    """The released data files are exactly what the seeded generator produces."""
    import gzip, json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "experiments" / "data" / "multipart"
    for name, spec in [("train", wide.TRAIN_SPEC), ("eval", wide.EVAL_SPEC)]:
        stored = [json.loads(l) for l in gzip.open(root / f"{name}.jsonl.gz", "rt")]
        assert stored == [json.loads(json.dumps(p, sort_keys=True)) for p in wide.make_set(**spec)]
