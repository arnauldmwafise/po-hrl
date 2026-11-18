"""Synthetic experiments for the PO-HRL spec (Section 6).

E1  structure: e(P), |J(P)|, search nodes, rounds, context and repair, by width
E2  signal density: zero-signal GRPO groups under outcome / chain-prefix / ideal rewards
E3  learning: credit assignment with a total order (chain) vs the partial order (ideal),
    plus flat GRPO and an ideal-bonus GRPO, on plans of increasing width; verifier noise
E4  attenuation check: measured chain/poset signal ratio vs Proposition P6

The learning task isolates credit assignment. Every subgoal g has a 4-way choice,
exactly one correct; g's output is correct iff its choice is correct and all its
prerequisites' outputs are correct. The answer is correct iff every subgoal is
correct, so the outcome does not depend on the plan's shape, only the credit does.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pohrl import Poset, chain, layered_random, search  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------- families
def family(name: str, rng: random.Random) -> Poset:
    """n = 10 elements including TOP."""
    if name == "chain":
        P = chain(9)
        return Poset(list(P.elements) + ["TOP"], {**{g: P.parents(g) for g in P.elements}, "TOP": ["g8"]})
    if name == "w2":
        return layered_random([2, 2, 2, 2, 1], rng, max_parents=2)
    if name == "w3":
        return layered_random([3, 3, 3], rng, max_parents=2)
    if name == "w5":
        return layered_random([5, 4], rng, max_parents=2)
    if name == "w9":
        return layered_random([9], rng)
    raise ValueError(name)


FAMILIES = ["chain", "w2", "w3", "w5", "w9"]


def family16(name: str, rng: random.Random) -> Poset:
    """Hard regime: 15 subgoals + TOP."""
    if name == "chain":
        P = chain(15)
        return Poset(list(P.elements) + ["TOP"], {**{g: P.parents(g) for g in P.elements}, "TOP": ["g14"]})
    sizes = {"w2": [2] * 7 + [1], "w3": [3] * 5, "w5": [5] * 3, "w8": [8, 7], "w15": [15]}[name]
    return layered_random(sizes, rng, max_parents=2)


FAMILIES16 = ["chain", "w2", "w3", "w5", "w8", "w15"]
HARD = dict(iters=300, G=8, lr=0.2, M=6, p_range=(0.3, 0.8))


def random_plan_n(n_inner: int, rng: random.Random) -> Poset:
    L = rng.randint(1, n_inner)
    cuts = sorted(rng.sample(range(1, n_inner), L - 1)) if L > 1 else []
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [n_inner])]
    return layered_random(sizes, rng, max_parents=rng.randint(1, 3))


# ----------------------------------------------------------------------- E1
def e1_structure(samples=400, seed=0):
    rng = random.Random(seed)
    rows = {}
    for _ in range(samples):
        P = random_plan_n(9, rng)
        w = P.width()
        order = P.sample_linear_extension(rng)
        sc1 = search.search_costs(P, 1)
        sc3 = search.search_costs(P, 3)
        cs = search.context_sizes(P, order)
        # repair: average over every non-TOP failure, done = everything not above it
        kp, kc = [], []
        for g in P.elements:
            if g == "TOP":
                continue
            r = search.repair_kept_fraction(P, order, g)
            kp.append(r["poset"]); kc.append(r["chain"])
        d = rows.setdefault(w, {"n": 0, "lin_ext": [], "ideals": [], "prefix_b1": [], "prefix_b3": [],
                                "idealdag_b3": [], "canon_b3": [], "height": [], "ctx_ratio": [],
                                "keep_poset": [], "keep_chain": []})
        d["n"] += 1
        d["lin_ext"].append(sc1["linear_extensions"]); d["ideals"].append(sc1["ideal_dag_nodes"])
        d["prefix_b1"].append(sc1["prefix_tree_nodes"]); d["prefix_b3"].append(sc3["prefix_tree_nodes"])
        d["idealdag_b3"].append(sc3["ideal_dag_nodes"]); d["canon_b3"].append(sc3["canonical_order_nodes"])
        d["height"].append(P.height())
        d["ctx_ratio"].append(cs["ideal_context"] / max(cs["sequential_history"], 1))
        d["keep_poset"].append(float(np.mean(kp))); d["keep_chain"].append(float(np.mean(kc)))
    summary = []
    for w in sorted(rows):
        d = rows[w]
        summary.append({
            "width": w, "plans": d["n"],
            "median_linear_extensions": float(np.median(d["lin_ext"])),
            "median_ideals": float(np.median(d["ideals"])),
            "median_prefix_nodes_b1": float(np.median(d["prefix_b1"])),
            "median_prefix_nodes_b3": float(np.median(d["prefix_b3"])),
            "median_ideal_dag_nodes_b3": float(np.median(d["idealdag_b3"])),
            "canonical_nodes_b3": float(np.median(d["canon_b3"])),
            "mean_rounds_height": float(np.mean(d["height"])),
            "mean_context_ratio": float(np.mean(d["ctx_ratio"])),
            "mean_kept_after_failure_poset": float(np.mean(d["keep_poset"])),
            "mean_kept_after_failure_chain": float(np.mean(d["keep_chain"])),
        })
    (OUT / "e1_structure.json").write_text(json.dumps(summary, indent=1))
    return summary


# ----------------------------------------------------------------------- E2
def correct_sets(P: Poset, p: np.ndarray, G: int, nrng: np.random.Generator):
    """Sample G rollouts: own-choice correctness, then propagated correctness."""
    els = list(P.elements)
    own = nrng.random((G, len(els))) < p  # p per element
    idx = {g: i for i, g in enumerate(els)}
    down = [[idx[u] for u in P.strict_down(g)] for g in els]
    prop = own.copy()
    for j, d in enumerate(down):
        if d:
            prop[:, j] &= own[:, d].all(axis=1)
    return own, prop  # prop[:, j] = g_j in I* (verified ideal, noiseless verifier)


def e2_signal(seed=1, trials=4000, G=8):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = []
    for fam in ["chain", "w3", "w9"]:
        for p in [0.6, 0.7, 0.8, 0.9]:
            deg = {"outcome": 0, "chain_prefix": 0, "ideal": 0}
            for t in range(trials // 20):
                P = family(fam, rng)
                els = list(P.elements)
                order = P.sample_linear_extension(rng)
                pos = np.array([order.index(g) for g in els])
                for _ in range(20):
                    own, prop = correct_sets(P, np.full(len(els), p), G, nrng)
                    outcome = prop.all(axis=1).astype(float)
                    ideal = prop.mean(axis=1)
                    # chain prefix: longest initial segment of `order` that is all correct (own)
                    own_by_pos = own[:, np.argsort(pos)]
                    first_bad = np.where(~own_by_pos, np.arange(len(els)), len(els)).min(axis=1)
                    chainp = first_bad / len(els)
                    deg["outcome"] += float(np.ptp(outcome) == 0)
                    deg["chain_prefix"] += float(np.ptp(chainp) == 0)
                    deg["ideal"] += float(np.ptp(ideal) == 0)
            n = (trials // 20) * 20
            out.append({"family": fam, "p": p, **{k: v / n for k, v in deg.items()}})
    (OUT / "e2_signal.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------- E3
def train(P: Poset, method: str, seed: int, iters=250, G=8, lr=0.4, M=4,
          noise=0.0, lam=0.5, p_range=(0.5, 0.9), norm=True):
    """Tabular softmax policy per subgoal; GRPO-style group-normalized advantages.

    methods:
      flat         outcome reward, same advantage for every decision
      ideal_bonus  outcome + lam * |I*|/n, same advantage for every decision (D4 only)
      chain        per-subgoal credit 1[g in verified prefix of a fixed linear extension]
      poset        per-subgoal credit 1[g in verified ideal I*] (D3 + D4, Prop. P5)
    returns the exact success probability (prod of p_g) after each iteration.
    """
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    els = list(P.elements)
    n = len(els)
    idx = {g: i for i, g in enumerate(els)}
    down = [[idx[u] for u in P.strict_down(g)] for g in els]
    order = P.sample_linear_extension(rng)
    before = [[idx[u] for u in order[: order.index(g)]] for g in els]
    # heterogeneous difficulty: initial prob of the correct choice in [0.5, 0.9]
    p0 = np.array([rng.uniform(*p_range) for _ in els])
    logits = np.zeros((n, M))
    logits[:, 0] = np.log(p0 * (M - 1) / (1 - p0))  # softmax gives p0 to action 0
    hist = []
    for _ in range(iters):
        pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        hist.append(float(np.prod(pr[:, 0])))
        cum = np.cumsum(pr, axis=1)
        acts = (nrng.random((G, n, 1)) > cum[None, :, :]).sum(-1).clip(0, M - 1)
        own = acts == 0
        prop = own.copy()
        for j, d in enumerate(down):
            if d:
                prop[:, j] &= own[:, d].all(axis=1)
        outcome = prop.all(axis=1).astype(float)
        if noise > 0:  # verifier flips each subgoal verdict with prob `noise` (not the final answer)
            flip = nrng.random(prop.shape) < noise
            seen_own = own ^ flip
        else:
            seen_own = own
        vprop = seen_own.copy()
        for j, d in enumerate(down):
            if d:
                vprop[:, j] &= seen_own[:, d].all(axis=1)
        if method in ("flat", "ideal_bonus"):
            R = outcome + (lam * vprop.mean(1) if method == "ideal_bonus" else 0.0)
            A = (R - R.mean()) / ((R.std() + 1e-6) if norm else 1.0)
            adv = np.repeat(A[:, None], n, axis=1)
        else:
            if method == "poset":
                c = vprop.astype(float)
            else:  # chain: g credited only if every element before it in the order is verified
                c = np.stack([seen_own[:, j] & seen_own[:, before[j]].all(axis=1) if before[j] else seen_own[:, j]
                              for j in range(n)], axis=1).astype(float)
            Ac = (c - c.mean(0)) / ((c.std(0) + 1e-6) if norm else 1.0)
            Ao = (outcome - outcome.mean()) / ((outcome.std() + 1e-6) if norm else 1.0)
            adv = Ac + Ao[:, None]
        onehot = np.zeros((G, n, M)); onehot[np.arange(G)[:, None], np.arange(n)[None, :], acts] = 1.0
        grad = (adv[:, :, None] * (onehot - pr[None])).sum(0)
        logits += lr * grad / G
    return hist


def first_reach(curve, thr=0.5):
    for i, v in enumerate(curve):
        if v >= thr:
            return i
    return None


def e3_learning(seeds=24):
    """Easy regime (n=10, p0 in [0.5,0.9], M=4) and hard regime (n=16, p0 in [0.3,0.8], M=6)."""
    res = {"easy": {}, "hard": {}, "summary": []}
    for regime, fams, fam_fn, kw in [("easy", FAMILIES, family, dict(iters=250)),
                                     ("hard", FAMILIES16, family16, HARD)]:
        for fam in fams:
            for method in ["flat", "ideal_bonus", "chain", "poset"]:
                curves = [train(fam_fn(fam, random.Random(1000 + s)), method, seed=s, **kw) for s in range(seeds)]
                mean = np.mean(curves, axis=0)
                res[regime][f"{fam}|{method}"] = mean.tolist()
                reach = [first_reach(c) for c in curves]
                res["summary"].append({
                    "regime": regime, "family": fam, "method": method,
                    "success_at_50": float(mean[50]), "success_at_100": float(mean[100]),
                    "final": float(mean[-1]),
                    "median_iters_to_50pct": (float(np.median([r for r in reach if r is not None]))
                                              if all(r is not None for r in reach) else None),
                    "seeds_reaching_50pct": sum(r is not None for r in reach),
                    "per_seed_iters_to_50pct": reach,
                    "per_seed_final": [float(c[-1]) for c in curves],
                })
    (OUT / "e3_learning.json").write_text(json.dumps(res))
    return res


def e3_noise(seeds=24):
    res = {"curves": {}, "summary": []}
    for noise in [0.0, 0.1, 0.2, 0.3]:
        for method in ["chain", "poset"]:
            curves = [train(family16("w5", random.Random(2000 + s)), method, seed=s, noise=noise, **HARD)
                      for s in range(seeds)]
            mean = np.mean(curves, axis=0)
            res["curves"][f"{noise}|{method}"] = mean.tolist()
            res["summary"].append({"noise": noise, "method": method, "success_at_50": float(mean[50]),
                                   "success_at_100": float(mean[100]), "final": float(mean[-1]),
                                   "per_seed_final": [float(c[-1]) for c in curves]})
    (OUT / "e3_noise.json").write_text(json.dumps(res))
    return res


# ----------------------------------------------------------------------- E4
def e4_attenuation(seed=3, samples=300, p=0.75):
    """Prop. P6: expected chain credit / poset credit for g = p^(# incomparable elements before g)."""
    rng = random.Random(seed)
    pred, meas = [], []
    nrng = np.random.default_rng(seed)
    for _ in range(samples):
        P = family(rng.choice(["w2", "w3", "w5", "w9"]), rng)
        els = list(P.elements)
        order = P.sample_linear_extension(rng)
        own, prop = correct_sets(P, np.full(len(els), p), 4000, nrng)
        for j, g in enumerate(els):
            bef = [els.index(u) for u in order[: order.index(g)]]
            k = sum(1 for u in order[: order.index(g)] if not P.comparable(u, g))
            chain_c = (own[:, j] & own[:, bef].all(axis=1)).mean() if bef else own[:, j].mean()
            pos_c = prop[:, j].mean()
            if pos_c > 0.05:
                pred.append(p ** k); meas.append(chain_c / pos_c)
    err = float(np.mean(np.abs(np.array(pred) - np.array(meas))))
    # Corollary: for an antichain of k subgoals under a uniform order, E[A] = (1 - p^k) / (k (1 - p))
    anti = []
    for k in [2, 4, 8, 16]:
        P = Poset([f"g{i}" for i in range(k)] + ["TOP"], {"TOP": [f"g{i}" for i in range(k)]})
        # genuine Monte Carlo: random order, random subgoal, sampled step successes
        T = 40000
        chain_sum = poset_sum = 0.0
        fP = P.count_completions()
        for _ in range(T):
            order = P.sample_linear_extension(rng, fP)
            g = f"g{rng.randrange(k)}"
            ok = {u: rng.random() < p for u in P.elements}
            poset_sum += ok[g]                                    # g has no prerequisites
            chain_sum += ok[g] and all(ok[u] for u in order[: order.index(g)])
        anti.append({"k": k, "simulated": chain_sum / poset_sum, "closed_form": (1 - p ** k) / (k * (1 - p)),
                     "trials": T})
    out = {"pairs": len(pred), "mean_abs_error": err,
           "mean_predicted_ratio": float(np.mean(pred)), "mean_measured_ratio": float(np.mean(meas)),
           "pairs_predicted": [float(x) for x in pred], "pairs_measured": [float(x) for x in meas],
           "antichain": anti}
    (OUT / "e4_attenuation.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4", "noise"]
    if "e1" in which: print(json.dumps(e1_structure(), indent=1))
    if "e2" in which: print(json.dumps(e2_signal(), indent=1))
    if "e4" in which: print(json.dumps(e4_attenuation(), indent=1))
    if "e3" in which:
        for row in e3_learning()["summary"]: print(row)
    if "noise" in which:
        for row in e3_noise()["summary"]: print(row)
