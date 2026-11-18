"""Record exactly which real problems the language-model experiments used.

The experiments download MATH, MATH500 and AIME 2024 from the Hugging Face Hub and select problems with fixed
seeds (see ``scripts/cloud/job.py`` and ``notebooks/M2_inference.ipynb``). This script repeats those selections
and writes a manifest that identifies every problem by its source data set, configuration, split and row index,
together with SHA-256 checksums of the problem text and answer. The manifest stores no problem text, so it can be
redistributed with the code; the problems themselves are fetched from the Hub with ``load_problems``.

Usage (needs internet access and ``pip install datasets``; no GPU):

    python experiments/llm_data.py                      # write experiments/data/llm/*.jsonl and meta.json
    python experiments/llm_data.py --check              # compare a written manifest with the Hub and the stored results

Files written to ``experiments/data/llm/``:

    m3_train.jsonl   1,200 M3 training problems, ids tr0..tr1199 as in experiments/results/m3/plans_train.jsonl.gz
    m3_eval.jsonl    367 MATH500 problems of levels 3-5 (M3 evaluation pool), ids as in plans_eval.jsonl.gz
    m2.jsonl         130 M2 problems (100 hardest MATH500 problems and AIME 2024), ids as in results/m2_v2
    meta.json        data set revisions (Hub commit hashes), selection rules and counts
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pohrl_llm.data import last_boxed  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
OUT_DEFAULT = ROOT / "experiments" / "data" / "llm"

MATH = "EleutherAI/hendrycks_math"
MATH500 = "HuggingFaceH4/MATH-500"
AIME = "HuggingFaceH4/aime_2024"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]          # order used by scripts/cloud/job.py
N_TRAIN, N_M2_MATH = 1200, 100


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _answer(raw: str) -> str:
    # same rule as pohrl_llm.data.load_hf
    raw = str(raw)
    if "####" in raw:
        return raw.split("####")[-1].strip()
    if "\\boxed" in raw:
        return last_boxed(raw) or raw
    return raw


def _revision(name: str):
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(name).sha
    except Exception as e:  # pragma: no cover - network dependent
        return f"unknown ({type(e).__name__})"


def _rows(name: str, split: str, qkey: str, akey: str, config=None) -> List[Dict]:
    """Rows in the order and with the fields that pohrl_llm.data.load_hf produces, plus their provenance."""
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    out = []
    for i, r in enumerate(ds):
        row = {"id": str(r.get("unique_id", r.get("id", i))), "problem": r[qkey], "answer": _answer(r[akey]),
               "source": {"dataset": name, "config": config, "split": split, "index": i}}
        if "level" in r:
            row["level"] = r["level"]
        out.append(row)
    return out


def select() -> Dict[str, List[Dict]]:
    # M3 training set: MATH train, levels 3-5, subjects in fixed order, shuffled with seed 0, first 1,200
    keep = lambda r: str(r.get("level", "")) in ("Level 3", "Level 4", "Level 5")
    train = []
    for s in SUBJECTS:
        train += [r for r in _rows(MATH, "train", "problem", "solution", config=s) if keep(r)]
    random.Random(0).shuffle(train)
    train = train[:N_TRAIN]
    for i, r in enumerate(train):
        r["pid"] = f"tr{i}"
    # M3 evaluation pool: MATH500 levels 3-5
    m500 = _rows(MATH500, "test", "problem", "answer")
    ev = [dict(r, pid=r["id"]) for r in m500 if int(r.get("level", 0)) >= 3]
    # M2: MATH500 shuffled with seed 0, stable-sorted by level, hardest 100; then all of AIME 2024
    m2_math = list(m500)
    random.Random(0).shuffle(m2_math)
    m2_math = sorted(m2_math, key=lambda r: r.get("level", 0))[-N_M2_MATH:]
    aime = _rows(AIME, "train", "problem", "answer")
    m2 = [dict(r, pid=r["id"]) for r in m2_math] + [dict(r, pid="aime24-" + r["id"]) for r in aime]
    return {"m3_train": train, "m3_eval": ev, "m2": m2}


def manifest_row(r: Dict) -> Dict:
    return {"pid": r["pid"], **r["source"], "source_id": r["id"], "level": r.get("level"),
            "problem_sha256": sha(r["problem"]), "answer_sha256": sha(r["answer"])}


def stored_ids() -> Dict[str, List[str]]:
    """Problem ids recorded in the result files of the paper, in their stored order."""
    def gz(p):
        return [json.loads(l)["pid"] for l in gzip.open(p, "rt")]
    return {"m3_train": gz(RESULTS / "m3" / "plans_train.jsonl.gz"),
            "m3_eval": gz(RESULTS / "m3" / "plans_eval.jsonl.gz"),
            "m2": [json.loads(l)["pid"] for l in open(RESULTS / "m2_v2" / "m2_plans.jsonl")]}


def compare_with_results(sel_ids: Dict[str, List[str]]) -> List[str]:
    """Compare selected ids with the ids stored in the result files."""
    msgs = []
    st = stored_ids()
    for k in sel_ids:
        a, b = set(sel_ids[k]), set(st[k])
        if a != b:
            msgs.append(f"{k}: MISMATCH, {len(b - a)} stored ids not selected, {len(a - b)} selected ids not stored")
        elif k == "m3_train":
            msgs.append(f"{k}: {len(a)} problems (ids tr0..tr{len(a) - 1} are positions, so this checks the count only; "
                        "see the text check below)")
        else:
            msgs.append(f"{k}: {len(a)} problems, ids identical to the stored results")
    return msgs


_LATEX = re.compile(r"\\[a-zA-Z]+")       # LaTeX commands such as \frac, \mathbf
_TOK = re.compile(r"[a-z]{3,}|\d+")


def _tokens(text: str) -> List[str]:
    return _TOK.findall(_LATEX.sub(" ", text).lower())


def plan_text_check(train: List[Dict], head: int = 300) -> str:
    """Evidence from the cached M3 training plans that tr<i> is the problem the selection assigns to it.

    The planner's reasoning begins by restating its problem in plain words, so the first ``head`` characters of the
    plan for tr<i> are compared (TF-IDF cosine similarity, LaTeX commands removed) with all MATH training problems,
    about 7,500. For each plan we record the rank of the assigned problem among them (identical duplicates tie). The
    same ranks are computed for a random re-pairing of plans and problems, which shows what chance gives. Because the
    plans paraphrase, the assigned problem is not always ranked first; the selection is confirmed when it is in the
    top 20 for most plans (chance: 20 in 7,500) and the median rank is at most 5.
    """
    import collections
    import math
    plans = {json.loads(l)["pid"]: json.loads(l)["plan"] for l in gzip.open(RESULTS / "m3" / "plans_train.jsonl.gz", "rt")}
    pool = []
    for s in SUBJECTS:
        pool += _rows(MATH, "train", "problem", "solution", config=s)
    docs = [_tokens(r["problem"]) for r in pool]
    df = collections.Counter(w for d in docs for w in set(d))
    idf = {w: math.log(len(pool) / c) for w, c in df.items()}

    def vec(ts):
        c = collections.Counter(w for w in ts if w in idf)
        v = {w: (1 + math.log(n)) * idf[w] for w, n in c.items()}
        z = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / z for w, x in v.items()}
    inv = collections.defaultdict(list)
    for j, d in enumerate(docs):
        for w, x in vec(d).items():
            inv[w].append((j, x))
    pos = {(r["source"]["config"], r["source"]["index"]): j for j, r in enumerate(pool)}

    def rank(pid, t):
        sc = collections.defaultdict(float)
        for w, x in vec(_tokens(plans[pid].replace("<think>", "")[:head])).items():
            for j, y in inv.get(w, ()):
                sc[j] += x * y
        st = sc.get(t, 0.0)
        return 1 + sum(1 for j, v in sc.items() if v > st and pool[j]["problem"] != pool[t]["problem"])
    pairs = [(r["pid"], pos[(r["source"]["config"], r["source"]["index"])]) for r in train if r["pid"] in plans]
    targets = [t for _, t in pairs]
    random.Random(1).shuffle(targets)
    true = [rank(p, t) for p, t in pairs]
    rand = [rank(p, t) for (p, _), t in zip(pairs, targets)]
    share = lambda rr, k: 100.0 * sum(x <= k for x in rr) / max(len(rr), 1)
    med = lambda rr: sorted(rr)[len(rr) // 2] if rr else 0
    ok = share(true, 20) >= 80 and med(true) <= 5 and share(rand, 20) < 5
    return (f"m3_train text check ({len(pairs)} cached plans, {len(pool)} MATH training problems): the assigned problem "
            f"is the best match for {share(true, 1):.1f}% of plans and in the top 20 for {share(true, 20):.1f}% "
            f"(median rank {med(true)}); for a random pairing, {share(rand, 1):.1f}% and {share(rand, 20):.1f}% "
            f"(median rank {med(rand)}) ({'consistent' if ok else 'NOT consistent'} with the selection)")


def write(out: Path) -> None:
    sel = select()
    out.mkdir(parents=True, exist_ok=True)
    for k, rows in sel.items():
        with open(out / f"{k}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(manifest_row(r), sort_keys=True) + "\n")
    report = compare_with_results({k: [r["pid"] for r in v] for k, v in sel.items()})
    report.append(plan_text_check(sel["m3_train"]))
    meta = {"revisions": {n: _revision(n) for n in (MATH, MATH500, AIME)},
            "counts": {k: len(v) for k, v in sel.items()},
            "rules": {"m3_train": f"{MATH} train, configs {SUBJECTS} in this order, levels 3-5, "
                                  f"random.Random(0).shuffle, first {N_TRAIN}, ids tr0..tr{N_TRAIN - 1}",
                      "m3_eval": f"{MATH500} test, level >= 3",
                      "m2": f"{MATH500} test, random.Random(0).shuffle, stable sort by level, last {N_M2_MATH}; "
                            f"then all of {AIME} train with ids prefixed 'aime24-'"},
            "check_against_results": report}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("\n".join(report))
    print(f"manifest written to {out}")


def load_problems(manifest: Path) -> List[Dict]:
    """Fetch the problems listed in a manifest from the Hub and verify them against its checksums."""
    from datasets import load_dataset
    rows = [json.loads(l) for l in open(manifest)]
    cache, out = {}, []
    for m in rows:
        key = (m["dataset"], m["config"], m["split"])
        if key not in cache:
            cache[key] = load_dataset(m["dataset"], m["config"], split=m["split"]) if m["config"] \
                else load_dataset(m["dataset"], split=m["split"])
        r = cache[key][m["index"]]
        problem = r["problem"]
        answer = _answer(r["solution"] if "solution" in r and m["dataset"] == MATH else r["answer"])
        if sha(problem) != m["problem_sha256"] or sha(answer) != m["answer_sha256"]:
            raise ValueError(f"{m['pid']}: the Hub row differs from the manifest (data set revised?)")
        out.append({"id": m["pid"], "problem": problem, "answer": answer, "level": m["level"]})
    return out


def check(out: Path) -> None:
    for k in ("m3_train", "m3_eval", "m2"):
        probs = load_problems(out / f"{k}.jsonl")
        print(f"{k}: {len(probs)} problems fetched and verified against the manifest checksums")
    print("\n".join(compare_with_results({k: [json.loads(l)["pid"] for l in open(out / f"{k}.jsonl")]
                                          for k in ("m3_train", "m3_eval", "m2")})))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--check", action="store_true", help="verify an existing manifest instead of writing one")
    a = ap.parse_args()
    (check if a.check else write)(Path(a.out))
