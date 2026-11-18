import random

import pytest

from pohrl import Poset, credit
from pohrl.hierarchy import is_valid_refinement, owen_bruteforce, quotient


def fine_example():
    # coarse plan: X -> Z, Y -> Z ; X refined into x1,x2 (independent), Y into y1->y2, Z into z
    Q = Poset(["x1", "x2", "y1", "y2", "z"], {"y2": ["y1"], "z": ["x1", "x2", "y2"]})
    block = {"x1": "X", "x2": "X", "y1": "Y", "y2": "Y", "z": "Z"}
    return Q, block


def test_quotient_and_refinement():
    Q, block = fine_example()
    P = quotient(Q, block)
    assert P.parents("Z") == {"X", "Y"} and len(P) == 3
    assert is_valid_refinement(Q, block)


@pytest.mark.parametrize("seed", range(20))
def test_owen_quotient_property(seed):
    """Proposition P8: block sums of Owen credits = coarse precedence Shapley."""
    Q, block = fine_example()
    P = quotient(Q, block)
    rng = random.Random(seed)
    table = {J: rng.random() for J in Q.ideals()}
    VQ = lambda J: table[J]
    inv = lambda I: frozenset(g for g in Q.elements if block[g] in I)
    VP = lambda I: table[inv(I)]
    ow = owen_bruteforce(Q, block, VQ)
    coarse = credit.shapley_exact(P, VP)
    for B in P.elements:
        assert sum(ow[g] for g in Q.elements if block[g] == B) == pytest.approx(coarse[B], abs=1e-9)


def test_fine_shapley_breaks_quotient_property():
    """Plain fine Shapley does not aggregate to the coarse credit in general."""
    Q, block = fine_example()
    P = quotient(Q, block)
    worst = 0.0
    for seed in range(50):
        rng = random.Random(100 + seed)
        table = {J: rng.random() for J in Q.ideals()}
        inv = lambda I: frozenset(g for g in Q.elements if block[g] in I)
        fine = credit.shapley_exact(Q, lambda J: table[J])
        coarse = credit.shapley_exact(P, lambda I: table[inv(I)])
        for B in P.elements:
            worst = max(worst, abs(sum(fine[g] for g in Q.elements if block[g] == B) - coarse[B]))
    assert worst > 1e-3
