"""The manifest script must select exactly the problems the experiments selected.

A fake ``datasets`` module serves small synthetic tables, so the test runs offline. The selections of
``experiments/llm_data.py`` are compared with the code the experiments ran: the training and evaluation selection
of ``scripts/cloud/job.py`` and the M2 selection of ``notebooks/M2_inference.ipynb``, both through
``pohrl_llm.data.load_hf``. If a manifest has been written, it is also checked against the stored results.
"""
import json
import random
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def _fake_tables():
    rng = random.Random(5)
    t = {}
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    for s in subjects:
        t[("EleutherAI/hendrycks_math", s, "train")] = [
            {"problem": f"{s} problem {i} with {rng.randint(3, 999)}", "level": f"Level {rng.randint(1, 5)}",
             "type": s, "solution": f"so \\boxed{{{rng.randint(0, 99)}}}"} for i in range(400)]
    t[("HuggingFaceH4/MATH-500", None, "test")] = [
        {"problem": f"m500 {i}", "answer": str(i % 7), "level": rng.randint(1, 5), "unique_id": f"test/x/{i}.json"}
        for i in range(500)]
    t[("HuggingFaceH4/aime_2024", None, "train")] = [
        {"problem": f"aime {i}", "answer": str(i), "id": 60 + i} for i in range(30)]
    return t


@pytest.fixture
def fake_datasets(monkeypatch):
    tables = _fake_tables()
    mod = types.ModuleType("datasets")

    def load_dataset(name, config=None, split=None):
        return tables[(name, config, split)]
    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)
    return tables


def _reference_selection():
    """The selection code as it ran in the experiments (copied from job.py and the M2 notebook)."""
    from pohrl_llm import data as D
    SUBJECTS = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
                'number_theory', 'prealgebra', 'precalculus']
    keep = lambda r: str(r.get('level', '')) in ('Level 3', 'Level 4', 'Level 5')
    train = []
    for s in SUBJECTS:
        train += D.load_hf('EleutherAI/hendrycks_math', split='train', qkey='problem', akey='solution', config=s,
                           filter_fn=keep)
    random.Random(0).shuffle(train); train = train[:1200]
    for i, r in enumerate(train): r['id'] = f'tr{i}'
    ev_pool = [r for r in D.load_hf('math500') if int(r.get('level', 0)) >= 3]
    math500 = D.load_hf('math500')
    random.Random(0).shuffle(math500)
    math500 = sorted(math500, key=lambda r: r.get('level', 0))[-100:]
    aime = D.load_hf('aime24')
    for r in aime: r['id'] = 'aime24-' + r['id']
    return {"m3_train": train, "m3_eval": ev_pool, "m2": math500 + aime}


def test_selection_matches_experiment_code(fake_datasets):
    import llm_data as L
    ref, sel = _reference_selection(), L.select()
    for k in ref:
        assert [r["id"] for r in ref[k]] == [r["pid"] for r in sel[k]], k
        assert [(r["problem"], r["answer"]) for r in ref[k]] == [(r["problem"], r["answer"]) for r in sel[k]], k


def test_manifest_round_trip(fake_datasets, tmp_path):
    import llm_data as L
    sel = L.select()
    for k, rows in sel.items():
        p = tmp_path / f"{k}.jsonl"
        p.write_text("".join(json.dumps(L.manifest_row(r)) + "\n" for r in rows))
        back = L.load_problems(p)
        assert [b["id"] for b in back] == [r["pid"] for r in rows]
        assert [b["problem"] for b in back] == [r["problem"] for r in rows]


def test_written_manifest_matches_results():
    import llm_data as L
    d = ROOT / "experiments" / "data" / "llm"
    if not (d / "m3_train.jsonl").exists():
        pytest.skip("manifest not written yet (run experiments/llm_data.py with internet access)")
    st = L.stored_ids()
    for k in ("m3_train", "m3_eval", "m2"):
        ids = [json.loads(l)["pid"] for l in open(d / f"{k}.jsonl")]
        assert set(ids) == set(st[k]), k
    meta = json.loads((d / "meta.json").read_text())
    assert not any("MISMATCH" in m or "NOT consistent" in m for m in meta["check_against_results"])
    assert any(m.startswith("m3_train text check") for m in meta["check_against_results"])
