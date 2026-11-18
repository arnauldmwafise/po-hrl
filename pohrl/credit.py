"""Credit assignment on the ideal lattice.

* precedence-constrained Shapley values (Faigle & Kern 1992), exact via a
  dynamic program over ideals (Proposition P3) and by Monte Carlo;
* potential-based shaping with an ideal potential (Proposition P4);
* TD residuals / GAE on a trajectory of ideals ("lattice HiPER");
* trace-class grouping of rollouts (partial-order reduction for GRPO groups).
"""
from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Callable, Dict, FrozenSet, Hashable, List, Sequence, Tuple

from .poset import Ideal, Poset

Value = Callable[[Ideal], float]


def shapley_exact(P: Poset, V: Value) -> Dict[Hashable, float]:
    """phi_g = sum_{J ideal, g in min(P\\J)} e(J) f(J+g) / e(P) * (V(J+g) - V(J)).

    e(J): linear extensions of J; f(K): linear extensions of P \\ K.
    The weight e(J) f(J+g)/e(P) is the probability that, in a uniformly random
    linear extension, the elements placed before g are exactly J.
    Cost: O(|J(P)| * w(P)) value calls.
    """
    e = P.count_extensions_below()
    f = P.count_completions()
    total = f[frozenset()]
    phi = {g: 0.0 for g in P.elements}
    for J in e:
        vJ = V(J)
        for g in P.frontier(J):
            K = J | {g}
            phi[g] += e[J] * f[K] / total * (V(K) - vJ)
    return phi


def shapley_bruteforce(P: Poset, V: Value) -> Dict[Hashable, float]:
    """Average marginal contribution over all linear extensions (enumerates permutations)."""
    phi = {g: 0.0 for g in P.elements}
    count = 0
    for perm in itertools.permutations(P.elements):
        if not P.is_linear_extension(list(perm)):
            continue
        count += 1
        J: FrozenSet = frozenset()
        for g in perm:
            K = J | {g}
            phi[g] += V(K) - V(J)
            J = K
    return {g: v / count for g, v in phi.items()}


def shapley_mc(P: Poset, V: Value, samples: int, rng: random.Random) -> Dict[Hashable, float]:
    f = P.count_completions()
    phi = {g: 0.0 for g in P.elements}
    for _ in range(samples):
        J: FrozenSet = frozenset()
        for g in P.sample_linear_extension(rng, f):
            K = J | {g}
            phi[g] += V(K) - V(J)
            J = K
    return {g: v / samples for g, v in phi.items()}


# ------------------------------------------------------------------ shaping
def ideal_weight_potential(P: Poset, weights: Dict[Hashable, float] | None = None) -> Value:
    w = weights or {g: 1.0 for g in P.elements}
    tot = sum(w.values())
    return lambda I: sum(w[g] for g in I) / tot


def shaped_rewards(ideals: Sequence[Ideal], rewards: Sequence[float], phi: Value,
                   gamma: float = 1.0, terminal_zero: bool = True) -> List[float]:
    """r'_t = r_t + gamma * Phi(I_{t+1}) - Phi(I_t); Phi(terminal) := 0 if terminal_zero.

    ideals has length T+1 (I_0..I_T), rewards length T.
    With terminal_zero, the discounted shaped return differs from the original
    by the constant -Phi(I_0), so optimal policies are unchanged (Ng et al. 1999).
    """
    T = len(rewards)
    assert len(ideals) == T + 1
    pot = [phi(I) for I in ideals]
    if terminal_zero:
        pot[-1] = 0.0
    return [rewards[t] + gamma * pot[t + 1] - pot[t] for t in range(T)]


def lattice_gae(ideals: Sequence[Ideal], rewards: Sequence[float], V: Value,
                gamma: float = 1.0, lam: float = 0.95) -> List[float]:
    """GAE over a trajectory of ideals I_0..I_T (one step = one round)."""
    T = len(rewards)
    vals = [V(I) for I in ideals]
    vals[-1] = 0.0  # terminal
    adv = [0.0] * T
    run = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * vals[t + 1] - vals[t]
        run = delta + gamma * lam * run
        adv[t] = run
    return adv


# ------------------------------------------------------- trace equivalence
def trace_key(executed: Sequence[Tuple[Hashable, Hashable]]) -> FrozenSet:
    """Canonical key of a rollout = the SET of (subgoal, output-id) pairs.
    Rollouts that differ only in the order of independent subgoals share a key."""
    return frozenset(executed)


def group_by_trace(rollouts: Sequence[Sequence[Tuple[Hashable, Hashable]]]) -> Dict[FrozenSet, List[int]]:
    groups: Dict[FrozenSet, List[int]] = defaultdict(list)
    for i, r in enumerate(rollouts):
        groups[trace_key(r)].append(i)
    return dict(groups)


def class_balanced_baseline(rewards: Sequence[float], rollouts) -> List[float]:
    """GRPO baseline where each trace class counts once (duplicates do not
    inflate the group mean). Returns advantages r_i - mean over classes."""
    groups = group_by_trace(rollouts)
    class_means = [sum(rewards[i] for i in idx) / len(idx) for idx in groups.values()]
    b = sum(class_means) / len(class_means)
    return [r - b for r in rewards]
