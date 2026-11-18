"""Search, scheduling, repair and context metrics (directions D2, D5, D6)."""
from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence, Set

from .poset import Ideal, Poset


def sleep_set_dfs_nodes(P: Poset) -> int:
    """Nodes visited by depth-first exploration of all schedules with sleep sets
    (Godefroid), without state caching. Distinct subgoals never conflict, so all
    schedules form one Mazurkiewicz trace; sleep sets prune most re-orderings but
    still visit dead-end partial branches. Measured, not assumed."""
    count = 0

    def dfs(I: Ideal, sleep: Set[Hashable]):
        nonlocal count
        count += 1
        done: Set[Hashable] = set()
        for g in P.frontier(I):
            if g in sleep:
                continue
            # independent of everything already explored at this node
            dfs(I | {g}, (sleep | done) - {g})
            done.add(g)

    dfs(frozenset(), set())
    return count


def search_costs(P: Poset, b: int = 1) -> Dict[str, int]:
    """Exhaustive search cost for one fixed plan when every subgoal has b
    candidate outputs (b samples per node in tree search).

    * prefix_tree_nodes: naive tree over (which subgoal next, which output);
      = sum_J e(J) b^|J|  (Proposition P2 with branching)
    * ideal_dag_nodes: states keyed by (ideal, outputs of its elements);
      = sum_J b^|J|
    * canonical_order_nodes: one fixed linear extension (persistent set of size 1);
      = sum_k b^k, k = 0..n
    """
    e = P.count_extensions_below()
    n = len(P)
    out = {
        "linear_extensions": P.num_linear_extensions(),
        "prefix_tree_nodes": sum(c * b ** len(J) for J, c in e.items()),
        "ideal_dag_nodes": sum(b ** len(J) for J in e),
        "canonical_order_nodes": sum(b ** k for k in range(n + 1)),
    }
    if b == 1:
        out["sleep_set_nodes"] = sleep_set_dfs_nodes(P)
    return out


def schedule_rounds(P: Poset, K: int) -> int:
    """Rounds needed with K executors and critical-path (remaining-height) priority."""
    rh = P.remaining_height()
    I: Set[Hashable] = set()
    rounds = 0
    while len(I) < len(P):
        fr = sorted(P.frontier(I), key=lambda g: -rh[g])
        I |= set(fr[:K])
        rounds += 1
    return rounds


def repair_kept_fraction(P: Poset, order: Sequence[Hashable], failed: Hashable,
                         done: Optional[Set[Hashable]] = None) -> Dict[str, float]:
    """Fraction of already-finished work that survives a failure of `failed`.

    * poset (filter pruning): keep done \\ up(failed)
    * chain (total order `order`): keep only the done steps placed before `failed`
    * restart: keep nothing
    `done` defaults to everything except up(failed) that is placed before the
    point where the failure is discovered in the chain order.
    """
    pos = {g: i for i, g in enumerate(order)}
    if done is None:
        done = {g for g in P.elements if g != failed and not P.leq(failed, g)}
    if not done:
        return {"poset": 1.0, "chain": 1.0, "restart": 1.0}
    up = P.up(failed)
    keep_poset = done - up
    keep_chain = {g for g in done if pos[g] < pos[failed]}
    n = len(done)
    return {"poset": len(keep_poset) / n, "chain": len(keep_chain) / n, "restart": 0.0}


def context_sizes(P: Poset, order: Sequence[Hashable]) -> Dict[str, int]:
    """Total subgoal outputs read by executors: principal-ideal context vs full history."""
    pos = {g: i for i, g in enumerate(order)}
    return {
        "ideal_context": sum(len(P.strict_down(g)) for g in P.elements),
        "parents_only": sum(len(P.parents(g)) for g in P.elements),
        "sequential_history": sum(pos[g] for g in P.elements),
    }
