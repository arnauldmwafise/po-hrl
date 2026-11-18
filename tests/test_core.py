import itertools
import math
import random
from collections import Counter

import pytest

from pohrl import Poset, antichain_with_top, chain, layered_random
from pohrl import credit, plan, search


def toy():
    # A, B -> C ; C, D -> ans   (the proposal's worked example)
    return Poset(["A", "B", "D", "C", "ans"], {"C": ["A", "B"], "ans": ["C", "D"]})


def random_posets(k=25, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        L = rng.randint(1, 4)
        sizes = [rng.randint(1, 3) for _ in range(L)]
        out.append(layered_random(sizes, rng, max_parents=rng.randint(1, 3)))
    return out


def brute_ideals(P):
    els = P.elements
    return [frozenset(s) for r in range(len(els) + 1) for s in itertools.combinations(els, r) if P.is_ideal(s)]


def brute_linexts(P):
    return [p for p in itertools.permutations(P.elements) if P.is_linear_extension(list(p))]


def brute_antichain_width(P):
    best = 0
    for r in range(len(P.elements) + 1):
        for s in itertools.combinations(P.elements, r):
            if all(not P.comparable(a, b) for a, b in itertools.combinations(s, 2)):
                best = max(best, r)
    return best


# ---------------------------------------------------------------- toy numbers
def test_toy_numbers_match_proposal():
    P = toy()
    assert P.num_linear_extensions() == 8
    assert P.num_ideals() == 11
    assert P.prefix_tree_size() == 34
    assert P.height() == 3 and P.width() == 3
    assert P.up("A") == {"A", "C", "ans"}
    assert set(P.frontier(set())) == {"A", "B", "D"}


# -------------------------------------------------------- exact vs brute force
@pytest.mark.parametrize("P", random_posets())
def test_counts_against_bruteforce(P):
    if len(P) > 9:
        pytest.skip("too large for brute force")
    assert set(P.ideals()) == set(brute_ideals(P))
    assert P.num_linear_extensions() == len(brute_linexts(P))
    prefixes = {p[:k] for p in brute_linexts(P) for k in range(len(P) + 1)}
    assert P.prefix_tree_size() == len(prefixes)
    assert P.width() == brute_antichain_width(P)


@pytest.mark.parametrize("P", random_posets(12, seed=3))
def test_prefix_tree_with_branching(P):
    if len(P) > 7:
        pytest.skip("too large")
    b = 2
    # brute force: sequences of (subgoal, output) following a linear extension prefix
    nodes = set()
    for p in brute_linexts(P):
        for k in range(len(P) + 1):
            for outs in itertools.product(range(b), repeat=k):
                nodes.add(tuple(zip(p[:k], outs)))
    assert search.search_costs(P, b)["prefix_tree_nodes"] == len(nodes)
    # ideal DAG: distinct (set of (subgoal, output)) states
    states = {frozenset(n) for n in nodes}
    assert search.search_costs(P, b)["ideal_dag_nodes"] == len(states)


def test_uniform_sampler():
    P = toy()
    rng = random.Random(1)
    f = P.count_completions()
    c = Counter(tuple(P.sample_linear_extension(rng, f)) for _ in range(16000))
    assert len(c) == 8
    for v in c.values():
        assert abs(v / 16000 - 1 / 8) < 0.015


# ------------------------------------------------------------------ Shapley
@pytest.mark.parametrize("P", random_posets(15, seed=7))
def test_shapley_exact_equals_bruteforce(P):
    if len(P) > 8:
        pytest.skip("too large")
    rng = random.Random(len(P))
    table = {J: rng.random() for J in P.ideals()}
    V = lambda J: table[J]
    ex = credit.shapley_exact(P, V)
    bf = credit.shapley_bruteforce(P, V)
    for g in P.elements:
        assert ex[g] == pytest.approx(bf[g], abs=1e-9)
    # efficiency: credits sum to V(P) - V(empty)
    assert sum(ex.values()) == pytest.approx(table[frozenset(P.elements)] - table[frozenset()])


def test_shapley_mc_converges():
    P = toy()
    rng = random.Random(0)
    table = {J: rng.random() for J in P.ideals()}
    ex = credit.shapley_exact(P, lambda J: table[J])
    mc = credit.shapley_mc(P, lambda J: table[J], 20000, random.Random(2))
    for g in P.elements:
        assert mc[g] == pytest.approx(ex[g], abs=0.02)


@pytest.mark.parametrize("P", random_posets(10, seed=11))
def test_shapley_of_verified_ideal_value_is_local(P):
    """Proposition P5: for V(J) = w(J & I*), phi_g = w_g * [g in I*]."""
    rng = random.Random(5)
    w = {g: rng.uniform(0.5, 2) for g in P.elements}
    correct = {g for g in P.elements if rng.random() < 0.7}
    Istar = P.largest_ideal_within(correct)
    V = lambda J: sum(w[g] for g in J if g in Istar)
    phi = credit.shapley_exact(P, V)
    for g in P.elements:
        assert phi[g] == pytest.approx(w[g] if g in Istar else 0.0)


def test_largest_ideal_within():
    P = toy()
    assert P.largest_ideal_within({"A", "C", "D", "ans"}) == frozenset({"A", "D"})
    assert P.largest_ideal_within(set(P.elements)) == frozenset(P.elements)


# ------------------------------------------------------------------ shaping
def test_potential_shaping_telescopes():
    P = toy()
    phi = credit.ideal_weight_potential(P)
    traj = [frozenset(), frozenset({"A", "B", "D"}), frozenset({"A", "B", "D", "C"}), frozenset(P.elements)]
    r = [0.0, 0.0, 1.0]
    shaped = credit.shaped_rewards(traj, r, phi, gamma=1.0)
    assert sum(shaped) == pytest.approx(sum(r) - phi(traj[0]))
    adv = credit.lattice_gae(traj, r, lambda J: len(J) / 5, gamma=1.0, lam=1.0)
    assert adv[0] == pytest.approx(sum(r) - 0.0)


def test_trace_classes():
    r1 = [("A", 0), ("B", 1), ("C", 0)]
    r2 = [("B", 1), ("A", 0), ("C", 0)]
    r3 = [("A", 1), ("B", 1), ("C", 0)]
    g = credit.group_by_trace([r1, r2, r3])
    assert sorted(len(v) for v in g.values()) == [1, 2]


# ------------------------------------------------------------- search/sched
def test_rounds_and_context():
    P = toy()
    assert search.schedule_rounds(P, K=10) == P.height() == 3
    assert search.schedule_rounds(P, K=1) == 5
    cs = search.context_sizes(P, ["A", "B", "D", "C", "ans"])
    assert cs["ideal_context"] == 2 + 4  # C reads A,B ; ans reads A,B,C,D
    assert cs["sequential_history"] == 0 + 1 + 2 + 3 + 4


def test_chain_has_no_savings():
    P = chain(6)
    c = search.search_costs(P, b=1)
    assert c["linear_extensions"] == 1 and c["prefix_tree_nodes"] == c["ideal_dag_nodes"] == 7
    assert P.width() == 1 and P.height() == 6


def test_antichain_counts():
    P = antichain_with_top(5)
    assert P.num_linear_extensions() == math.factorial(5)
    assert P.num_ideals() == 2 ** 5 + 1


def test_repair():
    P = toy()
    order = ["D", "A", "B", "C", "ans"]
    res = search.repair_kept_fraction(P, order, "A", done={"D", "B"})
    assert res["poset"] == 1.0 and res["chain"] == 0.5


# ---------------------------------------------------------------- plan parse
GOOD = """<plan>
<step id="s1" mode="compute" parents="">Sum of roots. CLAIM: 5</step>
<step id="s2" mode="compute" parents="">Product of roots. CLAIM: 6</step>
<step id="s3" mode="derive" parents="s1,s2">r1^2+r2^2. CLAIM: 13</step>
<step id="s4" mode="case" parents="">Residues mod 4.</step>
<step id="ans" mode="combine" parents="s3,s4,s1">Final count.</step>
</plan>"""


def test_plan_validation():
    r = plan.validate(GOOD)
    assert r.ok and not r.fallback
    assert r.poset.parents("ans") == {"s3", "s4"}  # s1 edge removed by V4
    assert r.steps["s3"].claim == "13"
    assert any("V4" in i for i in r.issues)


def test_plan_cycle_and_orphan():
    cyc = GOOD.replace('id="s1" mode="compute" parents=""', 'id="s1" mode="compute" parents="s3"')
    assert plan.validate(cyc).fallback
    orphan = GOOD.replace("</plan>", '<step id="s9" mode="recall" parents="">Unused.</step></plan>')
    r = plan.validate(orphan, orphan_policy="drop")
    assert r.ok and "s9" not in r.poset.elements and r.penalty < 0
    r = plan.validate(orphan, orphan_policy="link")
    assert r.ok and "s9" in r.poset.parents("ans") and r.penalty < 0
    assert any("linked" in i for i in r.issues)
    r = plan.validate(orphan)  # default 'next': s9 is last before nothing -> linked to ans
    assert r.ok and "s9" in r.poset.down("ans") and any("next-step" in i for i in r.issues)
    # the smoke-test failure: ans uses only a late step, earlier steps unlinked
    bad = """<plan>
<step id="s1" mode="compute" parents="">a. CLAIM: number</step>
<step id="s2" mode="compute" parents="">b. CLAIM: number</step>
<step id="s3" mode="combine" parents="s1,s2">c. CLAIM: number</step>
<step id="s4" mode="compute" parents="">d. CLAIM: number</step>
<step id="ans" mode="combine" parents="">final. CLAIM: number</step>
</plan>"""
    r = plan.validate(bad, orphan_policy="link")
    assert r.ok and r.poset.parents("ans") == {"s3", "s4"} and r.poset.width() == 3
    r = plan.validate(bad)  # conservative: s3 -> s4 (next written step), s4 -> ans
    assert r.ok and r.poset.parents("s4") == {"s3"} and r.poset.parents("ans") == {"s4"}
    assert r.poset.width() == 2 and set(r.poset.elements) == {"s1", "s2", "s3", "s4", "ans"}
    # last <plan> block wins (thinking text may contain a draft)
    r2 = plan.validate("<think><plan><step id=\"x\" mode=\"compute\" parents=\"\">draft</step></plan></think>" + GOOD)
    assert r2.ok and "s3" in r2.poset.elements


def test_revision_rule():
    old = plan.validate(GOOD).poset
    new = Poset(list(old.elements) + ["s5"], {**{g: old.parents(g) for g in old.elements}, "s5": ["s4"], "ans": ["s3", "s5"]})
    assert plan.revision_allowed(old, new, {"s1", "s2", "s4"})
    bad = Poset(list(old.elements), {**{g: old.parents(g) for g in old.elements}, "s1": ["s4"]})
    assert not plan.revision_allowed(old, bad, {"s1", "s2"})
