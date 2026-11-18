"""Synthetic multi-part math problems whose solution is a wide partial order (experiment M3-wide).

Each problem has k independent parts. Part X defines x_X from the problem's numbers and then y_X from
x_X; the answer is the sum of the y_X. The ground-truth plan is therefore k two-step chains joined at the
answer step: width k, height 3, and 4*C(k,2) incomparable pairs, i.e. iota(P)/|P| = 2k(k-1)/(2k+1),
which is 1.7 to 4.6 per step for k = 3..6 (the range of the hard tabular regime), against 0.35 for the
plans Qwen3-1.7B writes for MATH. Plans are given, not generated, so the experiment isolates executor
credit; every intermediate value is known exactly, so steps can be verified against gold sub-answers.
"""
from __future__ import annotations

import json
import math
import random
from typing import Dict, List

PARTS = "ABCDEF"


def _divisor_count(n: int) -> int:
    c, d = 1, 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d; e += 1
        c *= e + 1; d += 1
    return c * (2 if n > 1 else 1)


def _t_arith(r: random.Random, v: str):
    n, a, b, m = r.randint(15, 40), r.randint(2, 9), r.randint(-5, 12), r.choice([7, 11, 13, 17, 19, 23])
    x = sum(a * i + b for i in range(1, n + 1))
    return (f"Let $x_{v}$ be the sum $\\sum_{{i=1}}^{{{n}}} ({a}i {'+' if b >= 0 else '-'} {abs(b)})$, and let $y_{v}$ be the remainder when $x_{v}$ is divided by ${m}$.",
            f"Compute x_{v} = sum over i = 1..{n} of ({a}i {'+' if b >= 0 else '-'} {abs(b)}).",
            f"Compute y_{v} = remainder of x_{v} divided by {m}.", x, x % m)


def _t_divisors(r: random.Random, v: str):
    while True:
        N = r.choice([2, 4, 8, 16, 3, 9, 27]) * r.choice([3, 5, 9, 25, 7, 11]) * r.choice([1, 7, 11, 13, 17, 5])
        if 200 <= N <= 3000:
            break
    c, d = r.randint(2, 9), r.randint(1, 30)
    x = _divisor_count(N)
    return (f"Let $x_{v}$ be the number of positive divisors of ${N}$, and let $y_{v} = {c}x_{v} + {d}$.",
            f"Compute x_{v} = number of positive divisors of {N}.", f"Compute y_{v} = {c}*x_{v} + {d}.", x, c * x + d)


def _t_binom(r: random.Random, v: str):
    n, k, m = r.randint(8, 14), r.randint(2, 5), r.choice([7, 11, 13, 17, 19, 23, 29])
    x = math.comb(n, k)
    return (f"Let $x_{v} = \\binom{{{n}}}{{{k}}}$, and let $y_{v}$ be the remainder when $x_{v}$ is divided by ${m}$.",
            f"Compute x_{v} = C({n}, {k}).", f"Compute y_{v} = remainder of x_{v} divided by {m}.", x, x % m)


def _t_powmod(r: random.Random, v: str):
    b, e, m, c = r.randint(2, 12), r.randint(6, 15), r.choice([13, 17, 19, 23, 29, 31]), r.randint(1, 20)
    x = pow(b, e, m)
    return (f"Let $x_{v}$ be the remainder when ${b}^{{{e}}}$ is divided by ${m}$, and let $y_{v} = x_{v}^2 + {c}$.",
            f"Compute x_{v} = {b}^{e} mod {m}.", f"Compute y_{v} = x_{v}^2 + {c}.", x, x * x + c)


def _t_inclexcl(r: random.Random, v: str):
    N = r.randint(100, 600)
    p, q = r.choice([(3, 5), (3, 7), (4, 5), (5, 7), (2, 9), (4, 9), (5, 6), (6, 7)])
    x = N // p + N // q - N // math.lcm(p, q)
    return (f"Let $x_{v}$ be the number of integers from $1$ to ${N}$ that are divisible by ${p}$ or by ${q}$, and let $y_{v} = {N} - x_{v}$.",
            f"Compute x_{v} = count of integers in 1..{N} divisible by {p} or {q}.", f"Compute y_{v} = {N} - x_{v}.", x, N - x)


def _t_linear(r: random.Random, v: str):
    t, a, b = r.randint(-20, 40), r.randint(3, 15), r.randint(-50, 80)
    e, f = r.randint(2, 9), r.randint(1, 40)
    c = a * t + b
    return (f"Let $x_{v}$ be the solution of ${a}t {'+' if b >= 0 else '-'} {abs(b)} = {c}$, and let $y_{v} = {e}x_{v} - {f}$.",
            f"Compute x_{v} = the t that solves {a}t {'+' if b >= 0 else '-'} {abs(b)} = {c}.", f"Compute y_{v} = {e}*x_{v} - {f}.", t, e * t - f)


def _t_gcd(r: random.Random, v: str):
    while True:
        g, u, w = r.randint(6, 40), r.randint(5, 30), r.randint(5, 30)
        if math.gcd(u, w) == 1 and u != w:
            break
    a, b = g * u, g * w
    return (f"Let $x_{v} = \\gcd({a}, {b})$, and let $y_{v} = \\frac{{{a}}}{{x_{v}}} + \\frac{{{b}}}{{x_{v}}}$.",
            f"Compute x_{v} = gcd({a}, {b}).", f"Compute y_{v} = {a}/x_{v} + {b}/x_{v}.", g, u + w)


def _t_digits(r: random.Random, v: str):
    base, e, k = r.choice([2, 3, 7]), 0, r.randint(2, 6)
    e = {2: r.randint(12, 24), 3: r.randint(8, 15), 7: r.randint(5, 9)}[base]
    x = base ** e
    return (f"Let $x_{v} = {base}^{{{e}}}$, and let $y_{v}$ be {k} times the sum of the decimal digits of $x_{v}$.",
            f"Compute x_{v} = {base}^{e}.", f"Compute y_{v} = {k} * (sum of the decimal digits of x_{v}).", x, k * sum(map(int, str(x))))


def _t_roots(r: random.Random, v: str):
    while True:
        r1, r2 = r.randint(-12, 15), r.randint(-12, 15)
        if r1 != r2:
            break
    s, p, m = r1 + r2, r1 * r2, r.choice([7, 11, 13, 17, 19])
    poly = f"z^2 {'-' if s >= 0 else '+'} {abs(s)}z {'+' if p >= 0 else '-'} {abs(p)}"
    x = r1 * r1 + r2 * r2
    return (f"Let $x_{v}$ be the sum of the squares of the roots of ${poly} = 0$, and let $y_{v}$ be the remainder when $x_{v}$ is divided by ${m}$.",
            f"Compute x_{v} = sum of the squares of the roots of {poly} = 0.", f"Compute y_{v} = remainder of x_{v} divided by {m}.", x, x % m)


TEMPLATES = [_t_arith, _t_divisors, _t_binom, _t_powmod, _t_inclexcl, _t_linear, _t_gcd, _t_digits, _t_roots]


def make_problem(k: int, rng: random.Random, pid: str) -> Dict:
    """One problem with k independent two-step parts; returns problem text, answer, gold plan and sub-answers."""
    temps = rng.sample(TEMPLATES, k)
    texts, steps, sub, ys = [], [], {}, []
    for j, t in enumerate(temps):
        v = PARTS[j]
        text, d1, d2, x, y = t(rng, v)
        texts.append(f"Part {v}: {text}")
        s1, s2 = f"{v.lower()}1", f"{v.lower()}2"
        steps.append(f'<step id="{s1}" mode="compute" parents="">{d1} CLAIM: number</step>')
        steps.append(f'<step id="{s2}" mode="compute" parents="{s1}">{d2} CLAIM: number</step>')
        sub[s1], sub[s2] = str(x), str(y)
        ys.append(y)
    names = " + ".join(f"y_{PARTS[j]}" for j in range(k))
    answer = sum(ys)
    if k == 1:
        problem = f"{texts[0]} Compute $y_A$."
    else:
        problem = (f"This problem has {k} independent parts.\n" + "\n".join(texts) + f"\nCompute ${names}$.")
    parents = ",".join(f"{PARTS[j].lower()}2" for j in range(k))
    steps.append(f'<step id="ans" mode="combine" parents="{parents}">'
                 f'{"Report y_A." if k == 1 else "Add " + ", ".join(f"y_{PARTS[j]}" for j in range(k)) + "."} CLAIM: number</step>')
    plan = "<plan>\n" + "\n".join(steps) + "\n</plan>"
    sub["ans"] = str(answer)
    return {"id": pid, "problem": problem, "answer": str(answer), "sub_answers": sub, "plan": plan, "k": k}


def make_set(n: int, ks: List[int], seed: int, prefix: str) -> List[Dict]:
    rng = random.Random(seed)
    return [make_problem(ks[i % len(ks)], rng, f"{prefix}{i}") for i in range(n)]


def write_plan_cache(problems: List[Dict], path: str) -> None:
    """Write the gold plans in the format run_m3.cache_plans reads, so no planning call is made."""
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps({"pid": p["id"], "plan": p["plan"]}) + "\n")


def incomparable_pairs_per_step(k: int) -> float:
    return 4 * math.comb(k, 2) / (2 * k + 1)


TRAIN_SPEC = dict(n=1200, ks=[3, 4, 5, 6], seed=11, prefix="wtr")   # training set used in the paper
EVAL_SPEC = dict(n=240, ks=[1, 2, 3, 4, 5, 6], seed=12, prefix="wev")  # evaluation pool used in the paper


def write_set(problems: List[Dict], path: str) -> None:
    """Write full problem records (question, answer, sub-answers, gold plan, number of parts) as JSON lines."""
    import gzip
    import io
    text = "".join(json.dumps(p, sort_keys=True) + "\n" for p in problems)
    if path.endswith(".gz"):  # mtime=0 so the same problems always give byte-identical files
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            f.write(text.encode())
    else:
        with io.open(path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser(description="Generate the synthetic multi-part data set of the paper (deterministic).")
    ap.add_argument("--out", default="experiments/data/multipart")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, spec in [("train", TRAIN_SPEC), ("eval", EVAL_SPEC)]:
        probs = make_set(**spec)
        write_set(probs, os.path.join(a.out, f"{name}.jsonl.gz"))
        print(f"{name}: {len(probs)} problems -> {a.out}/{name}.jsonl.gz")
