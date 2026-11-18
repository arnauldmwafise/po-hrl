"""Multi-level plans as towers of order-preserving quotients, and Owen-style credit.

A fine plan Q is refined from a coarse plan P by a map pi: Q -> P that sends each
fine subgoal to its coarse block. If blocks are convex and pi is order preserving,
the preimage of every coarse ideal is a fine ideal, and block-contiguous orders
(coarse linear extension, then a linear extension inside each block) are linear
extensions of Q.

The precedence Owen value averages marginal contributions over block-contiguous
orders only. It satisfies the quotient property: the credits of a block sum to the
coarse precedence-Shapley credit of that block (Proposition P8), which plain fine
Shapley values do not in general.
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, FrozenSet, Hashable, List

from .poset import Poset


def quotient(Q: Poset, block_of: Dict[Hashable, Hashable]) -> Poset:
    blocks = list(dict.fromkeys(block_of[g] for g in Q.elements))
    parents = {B: set() for B in blocks}
    for v in Q.elements:
        for u in Q.parents(v):
            if block_of[u] != block_of[v]:
                parents[block_of[v]].add(block_of[u])
    return Poset(blocks, parents)


def is_valid_refinement(Q: Poset, block_of: Dict[Hashable, Hashable]) -> bool:
    try:
        P = quotient(Q, block_of)
    except ValueError:
        return False
    # convexity: u <= w <= v with u, v in B implies w in B
    members: Dict[Hashable, List[Hashable]] = {}
    for g in Q.elements:
        members.setdefault(block_of[g], []).append(g)
    for B, ms in members.items():
        for u in ms:
            for v in ms:
                for w in Q.elements:
                    if Q.lt(u, w) and Q.lt(w, v) and block_of[w] != B:
                        return False
    return True


def block_contiguous_orders(Q: Poset, block_of: Dict[Hashable, Hashable]):
    P = quotient(Q, block_of)
    members = {B: [g for g in Q.elements if block_of[g] == B] for B in P.elements}
    inner = {B: [p for p in itertools.permutations(ms) if Q.subposet(ms).is_linear_extension(list(p))]
             for B, ms in members.items()}
    for outer in itertools.permutations(P.elements):
        if not P.is_linear_extension(list(outer)):
            continue
        for combo in itertools.product(*(inner[B] for B in outer)):
            yield [g for part in combo for g in part]


def owen_bruteforce(Q: Poset, block_of, V: Callable[[FrozenSet], float]) -> Dict[Hashable, float]:
    phi = {g: 0.0 for g in Q.elements}
    n = 0
    for order in block_contiguous_orders(Q, block_of):
        n += 1
        J: FrozenSet = frozenset()
        for g in order:
            K = J | {g}
            phi[g] += V(K) - V(J)
            J = K
    return {g: v / n for g, v in phi.items()}
