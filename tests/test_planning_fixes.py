from pohrl import plan
from pohrl_llm import episode as EP
from pohrl_llm import prompts as PR

render = lambda m: PR.chat(m, None)
GOOD = """<plan>
<step id="s1" mode="compute" parents="">a. CLAIM: number</step>
<step id="s2" mode="compute" parents="s1">b. CLAIM: number</step>
<step id="s3" mode="compute" parents="">c. CLAIM: number</step>
<step id="ans" mode="combine" parents="s2,s3">final. CLAIM: number</step>
</plan>"""


def test_tolerant_parsing():
    stray = GOOD.replace('parents="s2,s3">', 'parents="s2,s3"">')
    assert plan.validate(stray).ok
    reordered = GOOD.replace('<step id="s3" mode="compute" parents="">', '<step parents="" id="s3" mode="compute">')
    assert plan.validate(reordered).ok
    unknown = GOOD.replace('parents="s1">', 'parents="s1,s9">')
    r = plan.validate(unknown)
    assert r.ok and r.poset.parents("s2") == {"s1"} and any("unknown parents dropped" in i for i in r.issues)
    big = "<plan>\n" + "\n".join(f'<step id="s{i}" mode="compute" parents="{"s%d" % (i - 1) if i > 1 else ""}">x</step>'
                                  for i in range(1, 14)) + '\n<step id="ans" mode="combine" parents="s13">f</step>\n</plan>'
    assert plan.validate(big).ok  # 14 steps now allowed (limit 16)


class ThinkingBackend:
    """First call: thinks until the budget runs out (no plan). Continuation: writes the plan body."""
    tokenizer = None

    def __init__(self):
        self.calls = []

    def generate(self, prompts, temperature=0.0, max_tokens=1024, lora=None):
        self.calls.append(prompts)
        out = []
        for p in prompts:
            if p.endswith("<plan>\n"):
                out.append(GOOD.split("<plan>\n", 1)[1])
            elif "check the dependencies" in p:
                out.append(GOOD.replace('parents="s1">', 'parents="">').replace('parents="s2,s3"', 'parents="s1,s2,s3"'))
            else:
                out.append("<think>let me just solve it... " * 50)
        return out


def test_budget_forcing_and_audit():
    be = ThinkingBackend()
    pp = EP.make_plans([{"id": "p", "problem": "Q", "answer": "1"}], be, render, audit=False)[0]
    assert pp.forced and pp.plan.ok and len(pp.plan.poset) == 4
    assert len(be.calls) == 2 and be.calls[1][0].endswith("</think>\n\n<plan>\n")
    be = ThinkingBackend()
    pp = EP.make_plans([{"id": "p", "problem": "Q", "answer": "1"}], be, render, audit=True)[0]
    assert pp.pre_audit is not None and pp.pre_audit.poset.parents("s2") == {"s1"}
    assert pp.plan.poset.parents("s2") == set() and pp.plan.poset.width() == 3


def test_last_step_renamed_to_ans():
    t = GOOD.replace('id="ans"', 'id="s4"')
    r = plan.validate(t)
    assert r.ok and "ans" in r.poset.elements and r.poset.parents("ans") == {"s2", "s3"}
    assert any("renamed to ans" in i for i in r.issues)


def test_m3_cache_and_stratify(tmp_path):
    from pohrl_llm import run_m3
    be = ThinkingBackend()
    ps = [{"id": f"p{i}", "problem": f"Q{i}", "answer": "1"} for i in range(5)]
    planned = run_m3.cache_plans(ps, be, render, str(tmp_path / "plans.jsonl"))
    assert all(pp.plan.ok and len(pp.plan.poset) == 4 for pp in planned)   # forced plans parsed
    again = run_m3.cache_plans(ps, be, render, str(tmp_path / "plans.jsonl"))
    assert len(be.calls) == 2                                              # cached: no new calls
    ev = run_m3.stratify(planned, per_bucket=3)
    assert len(ev) == 3 and all(run_m3.width_bucket(pp.plan.poset.width()) == "2-3" for pp in ev)
