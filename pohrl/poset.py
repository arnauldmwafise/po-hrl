"""Finite posets for PO-HRL plans.

A Poset is built from parent links (u -> v means u < v, "v uses u").
All order-theoretic quantities used by PO-HRL live here: down/up sets,
frontier (minimal elements of a complement), ideals, linear-extension
counts, exact uniform sampling of linear extensions, height and width.

Elements are hashable labels (strings in practice). Ideals are frozensets.
Everything is exact; costs scale with |J(P)| (number of ideals), which is
small for realistic plans (n <= ~20, modest width).
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Dict, FrozenSet, Hashable, Iterable, List, Optional, Set, Tuple

Elem = Hashable
Ideal = FrozenSet[Elem]


class CycleError(ValueError):
    pass


class Poset:
    def __init__(self, elements: Iterable[Elem], parents: Dict[Elem, Iterable[Elem]]):
        self.elements: Tuple[Elem, ...] = tuple(elements)
        eset = set(self.elements)
        if len(eset) != len(self.elements):
            raise ValueError("duplicate elements")
        self._parents: Dict[Elem, Set[Elem]] = {e: set() for e in self.elements}
        for v, ps in parents.items():
            if v not in eset:
                raise ValueError(f"unknown element {v!r}")
            for u in ps:
                if u not in eset:
                    raise ValueError(f"unknown parent {u!r} of {v!r}")
                if u == v:
                    raise CycleError(f"self loop at {v!r}")
                self._parents[v].add(u)
        self._down = self._compute_down()  # strict down-sets (transitive closure)
        self._up: Dict[Elem, Set[Elem]] = {e: set() for e in self.elements}
        for v in self.elements:
            for u in self._down[v]:
                self._up[u].add(v)
        self._index = {e: i for i, e in enumerate(self.elements)}

    # ---------------------------------------------------------------- basics
    def _compute_down(self) -> Dict[Elem, Set[Elem]]:
        down: Dict[Elem, Set[Elem]] = {}
        state: Dict[Elem, int] = {}

        def visit(v):
            st = state.get(v, 0)
            if st == 2:
                return
            if st == 1:
                raise CycleError(f"cycle through {v!r}")
            state[v] = 1
            acc: Set[Elem] = set()
            for u in self._parents[v]:
                visit(u)
                acc.add(u)
                acc |= down[u]
            down[v] = acc
            state[v] = 2

        for v in self.elements:
            visit(v)
        return down

    def __len__(self) -> int:
        return len(self.elements)

    def leq(self, a: Elem, b: Elem) -> bool:
        return a == b or a in self._down[b]

    def lt(self, a: Elem, b: Elem) -> bool:
        return a in self._down[b]

    def comparable(self, a: Elem, b: Elem) -> bool:
        return self.leq(a, b) or self.leq(b, a)

    def strict_down(self, g: Elem) -> Set[Elem]:
        return set(self._down[g])

    def down(self, g: Elem) -> Set[Elem]:
        return self._down[g] | {g}

    def up(self, g: Elem) -> Set[Elem]:
        """Principal filter: g and everything that depends on it."""
        return self._up[g] | {g}

    def parents(self, g: Elem) -> Set[Elem]:
        return set(self._parents[g])

    def covers(self) -> Dict[Elem, Set[Elem]]:
        """Hasse diagram: for each v, the elements it covers (transitive reduction)."""
        red: Dict[Elem, Set[Elem]] = {}
        for v in self.elements:
            below = self._down[v]
            red[v] = {u for u in below if not any(u in self._down[w] for w in below)}
        return red

    def redundant_edges(self) -> List[Tuple[Elem, Elem]]:
        cov = self.covers()
        return [(u, v) for v in self.elements for u in self._parents[v] if u not in cov[v]]

    def minimal(self, subset: Optional[Iterable[Elem]] = None) -> List[Elem]:
        s = set(self.elements) if subset is None else set(subset)
        return [g for g in self.elements if g in s and not (self._down[g] & s)]

    def maximal(self, subset: Optional[Iterable[Elem]] = None) -> List[Elem]:
        s = set(self.elements) if subset is None else set(subset)
        return [g for g in self.elements if g in s and not (self._up[g] & s)]

    def frontier(self, ideal: Iterable[Elem]) -> List[Elem]:
        """min(P \\ I): subgoals whose prerequisites are all in I."""
        I = set(ideal)
        return [g for g in self.elements if g not in I and self._down[g] <= I]

    def is_ideal(self, s: Iterable[Elem]) -> bool:
        S = set(s)
        return all(self._down[g] <= S for g in S)

    def is_filter(self, s: Iterable[Elem]) -> bool:
        S = set(s)
        return all(self._up[g] <= S for g in S)

    def largest_ideal_within(self, s: Iterable[Elem]) -> Ideal:
        """Largest ideal contained in s (well defined: unions of ideals are ideals)."""
        S = set(s)
        return frozenset(g for g in S if self._down[g] <= S)

    def ideal_closure(self, s: Iterable[Elem]) -> Ideal:
        out: Set[Elem] = set()
        for g in s:
            out |= self.down(g)
        return frozenset(out)

    def filter_closure(self, s: Iterable[Elem]) -> FrozenSet[Elem]:
        out: Set[Elem] = set()
        for g in s:
            out |= self.up(g)
        return frozenset(out)

    def top(self) -> Optional[Elem]:
        mx = self.maximal()
        return mx[0] if len(mx) == 1 else None

    def subposet(self, keep: Iterable[Elem]) -> "Poset":
        K = [e for e in self.elements if e in set(keep)]
        Ks = set(K)
        return Poset(K, {v: [u for u in self._down[v] if u in Ks] for v in K})

    # ------------------------------------------------------------ invariants
    def height(self) -> int:
        """Number of elements in a longest chain (= rounds with unlimited executors)."""
        memo: Dict[Elem, int] = {}

        def lvl(v):
            if v not in memo:
                memo[v] = 1 + max((lvl(u) for u in self._parents[v]), default=0)
            return memo[v]

        return max((lvl(v) for v in self.elements), default=0)

    def levels(self) -> Dict[Elem, int]:
        """Earliest round (1-based) each element can run in, with unlimited executors."""
        memo: Dict[Elem, int] = {}

        def lvl(v):
            if v not in memo:
                memo[v] = 1 + max((lvl(u) for u in self._parents[v]), default=0)
            return memo[v]

        return {v: lvl(v) for v in self.elements}

    def remaining_height(self) -> Dict[Elem, int]:
        """Longest chain starting at each element (critical-path priority)."""
        memo: Dict[Elem, int] = {}
        children: Dict[Elem, Set[Elem]] = {e: set() for e in self.elements}
        for v in self.elements:
            for u in self._parents[v]:
                children[u].add(v)

        def rh(v):
            if v not in memo:
                memo[v] = 1 + max((rh(c) for c in children[v]), default=0)
            return memo[v]

        return {v: rh(v) for v in self.elements}

    def width(self) -> int:
        """Size of a largest antichain, via Dilworth/Konig: n - max matching on comparabilities."""
        n = len(self.elements)
        idx = self._index
        adj = [[idx[v] for v in self._up[u]] for u in self.elements]
        match_r = [-1] * n

        def try_kuhn(u, seen):
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    if match_r[v] == -1 or try_kuhn(match_r[v], seen):
                        match_r[v] = u
                        return True
            return False

        m = 0
        for u in range(n):
            if try_kuhn(u, [False] * n):
                m += 1
        return n - m

    # -------------------------------------------------------------- ideals
    def ideals(self) -> List[Ideal]:
        """All order ideals (= all antichains, bijectively). BFS up the lattice."""
        start: Ideal = frozenset()
        seen = {start}
        order = [start]
        i = 0
        while i < len(order):
            I = order[i]
            i += 1
            for g in self.frontier(I):
                J = I | {g}
                if J not in seen:
                    seen.add(J)
                    order.append(J)
        return order

    def num_ideals(self) -> int:
        return len(self.ideals())

    # ---------------------------------------------------- linear extensions
    def count_extensions_below(self) -> Dict[Ideal, int]:
        """e(J) for every ideal J: number of linear extensions of the subposet J
        (= number of orders in which J can be completed starting from empty)."""
        ideals = sorted(self.ideals(), key=len)
        e: Dict[Ideal, int] = {frozenset(): 1}
        for J in ideals:
            if not J:
                continue
            # last element must be maximal in J
            e[J] = sum(e[J - {g}] for g in self.maximal(J))
        return e

    def count_completions(self) -> Dict[Ideal, int]:
        """f(J) for every ideal J: number of ways to order P \\ J (linear extensions of the filter P \\ J)."""
        ideals = sorted(self.ideals(), key=len, reverse=True)
        full = frozenset(self.elements)
        f: Dict[Ideal, int] = {full: 1}
        for J in ideals:
            if J == full:
                continue
            f[J] = sum(f[J | {g}] for g in self.frontier(J))
        return f

    def num_linear_extensions(self) -> int:
        return self.count_completions()[frozenset()]

    def sample_linear_extension(self, rng: random.Random, completions: Optional[Dict[Ideal, int]] = None) -> List[Elem]:
        """Exact uniform sample, using completion counts f(J)."""
        f = completions if completions is not None else self.count_completions()
        J: Ideal = frozenset()
        out: List[Elem] = []
        while len(out) < len(self.elements):
            cands = self.frontier(J)
            weights = [f[J | {g}] for g in cands]
            g = rng.choices(cands, weights=weights, k=1)[0]
            out.append(g)
            J = J | {g}
        return out

    def is_linear_extension(self, seq: List[Elem]) -> bool:
        if sorted(map(repr, seq)) != sorted(map(repr, self.elements)):
            return False
        pos = {g: i for i, g in enumerate(seq)}
        return all(pos[u] < pos[v] for v in self.elements for u in self._down[v])

    def prefix_tree_size(self) -> int:
        """Nodes in the prefix tree of all linear extensions (incl. root).
        Equals sum over ideals J of e(J) (Proposition P2)."""
        return sum(self.count_extensions_below().values())

    def __repr__(self) -> str:
        cov = self.covers()
        edges = ", ".join(f"{u}<{v}" for v in self.elements for u in sorted(cov[v], key=repr))
        return f"Poset(n={len(self)}, {edges})"


# ----------------------------------------------------------------- builders
def chain(n: int, prefix: str = "g") -> Poset:
    els = [f"{prefix}{i}" for i in range(n)]
    return Poset(els, {els[i]: [els[i - 1]] for i in range(1, n)})


def antichain_with_top(k: int) -> Poset:
    els = [f"g{i}" for i in range(k)] + ["TOP"]
    return Poset(els, {"TOP": els[:-1]})


def layered_random(layer_sizes: List[int], rng: random.Random, max_parents: int = 2,
                   top: bool = True) -> Poset:
    """Random layered plan: each node in layer l>0 gets 1..max_parents parents in layer l-1;
    every element without a child is joined to a single TOP element."""
    layers: List[List[str]] = []
    parents: Dict[str, List[str]] = {}
    c = 0
    for li, size in enumerate(layer_sizes):
        layer = []
        for _ in range(size):
            name = f"g{c}"
            c += 1
            layer.append(name)
            if li > 0:
                prev = layers[-1]
                k = rng.randint(1, min(max_parents, len(prev)))
                parents[name] = rng.sample(prev, k)
        layers.append(layer)
    els = [e for L in layers for e in L]
    if top:
        has_child = {u for ps in parents.values() for u in ps}
        sinks = [e for e in els if e not in has_child]
        els.append("TOP")
        parents["TOP"] = sinks
    return Poset(els, parents)
