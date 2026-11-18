"""P6 with a shared neural policy: a small transformer executor trained by RL.

Task. A problem is a random plan (poset) of n subgoals plus TOP. Every subgoal
computes op(a, b) mod M, where each operand is either a constant (if the subgoal
has fewer than two parents) or the OUTPUT the executor produced for a parent.
So an executor error propagates to everything above it, as in a derivation.
The executor is ONE transformer shared by all subgoals: input tokens
[op, a, b] -> a distribution over M digits.

Protocol.
 1. Supervised warm start on a fixed 30% of the arithmetic facts (op, a, b), so
    the policy starts competent on some facts and near chance on the rest.
 2. RL on whole problems: G rollouts per problem; each rollout executes the plan in
    a fixed linear extension (the "written order"), sampling every output.
    Methods differ ONLY in the per-subgoal advantage:
      flat   : group-normalized outcome (answer correct), same for every subgoal
      chain  : 1[g and every subgoal before it in the written order is correct]
      poset  : 1[g in the verified ideal]  (g and all its prerequisites correct)
      local  : 1[g's output is right GIVEN the outputs its parents produced]
               (a step-validity check; well defined because g reads only its parents)
    For chain/poset the outcome advantage is added, as in the spec (Section 4.1).
 3. Metric: greedy accuracy over the FULL fact table (every op, a, b), which
    measures what the shared policy has learned independently of plan shape,
    plus greedy whole-problem success on held-out problems.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pohrl import Poset, chain, layered_random  # noqa: E402

M = 11
OPS = ["add", "sub", "mul", "lin"]  # lin: 2a + b
NOP = len(OPS)
VOCAB = NOP + M  # op tokens then digit tokens


def apply_op(o, a, b):
    if o == 0: return (a + b) % M
    if o == 1: return (a - b) % M
    if o == 2: return (a * b) % M
    return (2 * a + b) % M


# --------------------------------------------------------------------- model
def init_params(key, d=64, layers=2, heads=4, mlp=128):
    ks = jax.random.split(key, 4 + 6 * layers)
    s = 0.02
    p = {"emb": jax.random.normal(ks[0], (VOCAB, d)) * s,
         "pos": jax.random.normal(ks[1], (3, d)) * s,
         "out": jax.random.normal(ks[2], (d, M)) * s,
         "ln_f": jnp.ones(d), "blocks": []}
    for l in range(layers):
        k = ks[4 + 6 * l: 10 + 6 * l]
        p["blocks"].append({
            "wq": jax.random.normal(k[0], (d, d)) * s, "wk": jax.random.normal(k[1], (d, d)) * s,
            "wv": jax.random.normal(k[2], (d, d)) * s, "wo": jax.random.normal(k[3], (d, d)) * s,
            "w1": jax.random.normal(k[4], (d, mlp)) * s, "w2": jax.random.normal(k[5], (mlp, d)) * s,
            "ln1": jnp.ones(d), "ln2": jnp.ones(d)})
    return p


def _ln(x, g):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return g * (x - mu) / jnp.sqrt(var + 1e-5)


def forward(p, x, heads=4):
    """x: int[N,3] -> logits[N,M] (read out at the last position)."""
    h = p["emb"][x] + p["pos"][None]
    N, T, d = h.shape
    hd = d // heads
    for b in p["blocks"]:
        z = _ln(h, b["ln1"])
        q = (z @ b["wq"]).reshape(N, T, heads, hd)
        k = (z @ b["wk"]).reshape(N, T, heads, hd)
        v = (z @ b["wv"]).reshape(N, T, heads, hd)
        att = jax.nn.softmax(jnp.einsum("nthd,nshd->nhts", q, k) / jnp.sqrt(hd), axis=-1)
        h = h + jnp.einsum("nhts,nshd->nthd", att, v).reshape(N, T, d) @ b["wo"]
        z = _ln(h, b["ln2"])
        h = h + jax.nn.gelu(z @ b["w1"]) @ b["w2"]
    return _ln(h, p["ln_f"])[:, -1] @ p["out"]


fwd = jax.jit(forward)


def tokens(o, a, b):
    return np.stack([o, NOP + a, NOP + b], axis=-1)


ALL_FACTS = np.array([(o, a, b) for o in range(NOP) for a in range(M) for b in range(M)])
ALL_X = tokens(ALL_FACTS[:, 0], ALL_FACTS[:, 1], ALL_FACTS[:, 2])
ALL_Y = np.array([apply_op(o, a, b) for o, a, b in ALL_FACTS])


def table_accuracy(p):
    return float((np.asarray(fwd(p, ALL_X)).argmax(-1) == ALL_Y).mean())


# ------------------------------------------------------------------ problems
def make_family(name, rng):
    if name == "chain":
        P = chain(15)
        return Poset(list(P.elements) + ["TOP"], {**{g: P.parents(g) for g in P.elements}, "TOP": ["g14"]})
    sizes = {"w3": [3] * 5, "w5": [5] * 3, "w15": [15]}[name]
    return layered_random(sizes, rng, max_parents=2)


class Problem:
    """Arrays over a fixed written order (a linear extension)."""

    def __init__(self, P: Poset, rng: random.Random):
        order = P.sample_linear_extension(rng)
        self.n = len(order)
        idx = {g: i for i, g in enumerate(order)}
        self.ops = np.array([rng.randrange(NOP) for _ in order])
        # operand sources: parent index (>=0) or -1 with a constant
        self.src = np.full((self.n, 2), -1)
        self.const = np.array([[rng.randrange(M), rng.randrange(M)] for _ in order])
        for i, g in enumerate(order):
            ps = sorted(idx[u] for u in P.parents(g))
            if len(ps) > 2:  # a binary op reads two parents; the answer still requires every subgoal
                ps = ps[-2:]
            for j, s in enumerate(ps):
                self.src[i, j] = s
        # prerequisite mask: down[i, j] = j in strict down-set of order[i] (through used parents)
        self.down = np.zeros((self.n, self.n), bool)
        for i in range(self.n):
            for j in self.src[i]:
                if j >= 0:
                    self.down[i, j] = True
                    self.down[i] |= self.down[j]
        self.truth = self.execute(None)
        self.width_label = None

    def operands(self, outs, i):
        a = outs[..., self.src[i, 0]] if self.src[i, 0] >= 0 else np.full(outs.shape[:-1], self.const[i, 0])
        b = outs[..., self.src[i, 1]] if self.src[i, 1] >= 0 else np.full(outs.shape[:-1], self.const[i, 1])
        return a, b

    def execute(self, sampler):
        outs = np.zeros(self.n, int)
        for i in range(self.n):
            a, b = self.operands(outs, i)
            outs[i] = apply_op(self.ops[i], int(a), int(b))
        return outs


def rollout(params, probs, G, key, greedy=False):
    """Execute every problem G times in its written order, batching across problems.
    Returns inputs X[B,G,n,3], outputs Y[B,G,n]."""
    B, n = len(probs), probs[0].n
    outs = np.zeros((B, G, n), int)
    X = np.zeros((B, G, n, 3), int)
    for i in range(n):
        A = np.zeros((B, G), int); Bv = np.zeros((B, G), int); O = np.zeros((B, G), int)
        for bi, pr in enumerate(probs):
            a, b = pr.operands(outs[bi], i)
            A[bi], Bv[bi], O[bi] = a, b, pr.ops[i]
        x = tokens(O, A, Bv).reshape(B * G, 3)
        logits = fwd(params, x)
        if greedy:
            y = np.asarray(logits.argmax(-1))
        else:
            key, sk = jax.random.split(key)
            y = np.asarray(jax.random.categorical(sk, logits))
        outs[:, :, i] = y.reshape(B, G)
        X[:, :, i] = x.reshape(B, G, 3)
    return X, outs, key


def advantages(probs, outs, method, X=None):
    B, G, n = outs.shape
    adv = np.zeros((B, G, n))
    for bi, pr in enumerate(probs):
        ok = outs[bi] == pr.truth[None]                       # g's output equals the true value
        poset = ok & np.all(ok[:, None, :] | ~pr.down[None], axis=2)   # g and all prerequisites
        outcome = poset.all(axis=1).astype(float)
        Ao = (outcome - outcome.mean()) / (outcome.std() + 1e-6)
        if method == "flat":
            adv[bi] = Ao[:, None]
            continue
        if method == "poset":
            c = poset.astype(float)
        elif method == "local":  # step valid given the outputs its parents actually produced
            x = X[bi]  # [G, n, 3] tokens: op, a, b
            local = np.vectorize(apply_op)(x[..., 0], x[..., 1] - NOP, x[..., 2] - NOP)
            c = (outs[bi] == local).astype(float)
        else:  # chain: everything before g in the written order must be correct
            c = np.cumprod(ok, axis=1).astype(float)
        Ac = (c - c.mean(0)) / (c.std(0) + 1e-6)
        adv[bi] = Ac + Ao[:, None]
    return adv


@partial(jax.jit, static_argnums=())
def pg_loss(p, x, y, a):
    logp = jax.nn.log_softmax(forward(p, x))
    return -(a * jnp.take_along_axis(logp, y[:, None], 1)[:, 0]).mean()


pg_grad = jax.jit(jax.value_and_grad(pg_loss))


def sft_loss(p, x, y):
    return optax.softmax_cross_entropy_with_integer_labels(forward(p, x), y).mean()


sft_grad = jax.jit(jax.value_and_grad(sft_loss))


def warm_start(seed, frac=0.3, steps=600, lr=3e-3):
    rng = np.random.default_rng(seed)
    params = init_params(jax.random.PRNGKey(seed))
    known = rng.random(len(ALL_FACTS)) < frac
    X, Y = ALL_X[known], ALL_Y[known]
    opt = optax.adam(lr); st = opt.init(params)
    for _ in range(steps):
        idx = rng.integers(0, len(X), 128)
        _, g = sft_grad(params, X[idx], Y[idx])
        up, st = opt.update(g, st); params = optax.apply_updates(params, up)
    return params, known


def run(family, method, seed, steps=300, B=16, G=8, lr=1e-3, warm=None, eval_every=25, log=None):
    rng = random.Random(seed)
    params, known = warm if warm is not None else warm_start(seed)
    pool = [Problem(make_family(family, rng), rng) for _ in range(400)]
    heldout = [Problem(make_family(family, rng), rng) for _ in range(200)]
    opt = optax.adam(lr); st = opt.init(params)
    key = jax.random.PRNGKey(10_000 + seed)
    hist = []

    def evaluate(step):
        _, outs, _ = rollout(params, heldout, 1, key, greedy=True)
        solved = float(np.mean([np.all(outs[i, 0] == heldout[i].truth) for i in range(len(heldout))]))
        acc = table_accuracy(params)
        pred = np.asarray(fwd(params, ALL_X)).argmax(-1) == ALL_Y
        hist.append({"step": step, "table_acc": acc, "unseen_acc": float(pred[~known].mean()),
                     "heldout_solved": solved})
        if log: log(f"{family} {method} s{seed} step {step}: table {acc:.3f} unseen {pred[~known].mean():.3f} solved {solved:.3f}")

    evaluate(0)
    for step in range(1, steps + 1):
        batch = rng.sample(pool, B)
        X, outs, key = rollout(params, batch, G, key)
        A = advantages(batch, outs, method, X)
        _, g = pg_grad(params, X.reshape(-1, 3), outs.reshape(-1), A.reshape(-1))
        up, st = opt.update(g, st); params = optax.apply_updates(params, up)
        if step % eval_every == 0:
            evaluate(step)
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="chain,w3,w5,w15")
    ap.add_argument("--methods", default="flat,chain,poset")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "transformer_p6.json"))
    a = ap.parse_args()
    out = Path(a.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    for seed in range(a.seeds):
        warm = warm_start(seed)  # same warm start for every method and family of this seed
        print(f"seed {seed}: warm-start table acc {table_accuracy(warm[0]):.3f}", flush=True)
        for fam in a.families.split(","):
            for m in a.methods.split(","):
                k = f"{fam}|{m}|{seed}"
                if k in results:
                    continue
                results[k] = run(fam, m, seed, steps=a.steps, warm=warm,
                                 log=lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True))
                out.write_text(json.dumps(results))
    print("done", time.time() - t0)
