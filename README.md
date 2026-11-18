# PO-HRL: Partial-Order Hierarchical Reinforcement Learning

Code, data, result logs and analysis scripts for

> **When Does Order Matter? Partial-Order Hierarchical Reinforcement Learning for Mathematical Reasoning in Language Models.**
> Arnauld Mesinga Mwafise, 2026.

Hierarchical reinforcement learning (HRL) methods for mathematical reasoning in large language models write
plans as sequences. PO-HRL represents a plan as a partially ordered set (poset) of subgoals instead, executes
independent subgoals in parallel, and assigns step-level credit through the lattice of order ideals. The paper
proves what a total order costs (attenuation by incomparable predecessors), shows that the total exposure is the
same for every order (it equals the number of incomparable pairs of the plan), gives exact precedence-constrained
Shapley credit on the ideal lattice, and tests the theory in simulations, with a shared transformer policy, and with
Qwen3-1.7B.

## Contents

| Path | What it holds |
| --- | --- |
| `pohrl/` | Order-theoretic core (pure Python): posets from parent links, ideals, frontier, height and width, exact linear-extension counting and uniform sampling, exact and Monte Carlo precedence Shapley credit on the ideal lattice, ideal-potential shaping, quotient posets and precedence Owen credit, search-cost formulas, plan parser and validator |
| `pohrl_llm/` | Language-model pipeline: prompts, answer checkers, round-based episode runner, step verification (gold, consensus, re-execution), credit rules (flat, chain, random-order chain, poset, local), LoRA trainer, M2/M3 runners, vLLM and scripted back ends, synthetic wide-plan problems (`wide.py`) |
| `experiments/sim.py` | Controlled experiments E1-E4: plan structure, reward signal density, learning with chain versus poset credit, verifier noise, attenuation law |
| `experiments/transformer_p6.py` | Shared transformer executor trained by RL (JAX, CPU): flat, chain and poset credit (a local-credit option is included for follow-up work and is not reported in the paper) |
| `experiments/m3_plan_stats.py` | Incomparable-pair statistics of the cached M3 plans (Proposition 4 and Section 7 of the paper) |
| `experiments/dagmath_stats.py` | Poset statistics of the DAG-Math gold dependency graphs |
| `experiments/llm_data.py` | Repeats the selection of the real problems used by M2 and M3 and writes a manifest of them (source data set, row, checksums) to `experiments/data/llm/`; `--check` fetches the listed problems and verifies them |
| `experiments/data/multipart/` | The synthetic multi-part data set of the paper (1,200 training and 240 evaluation problems with gold plans and sub-answers), regenerated exactly by `python -m pohrl_llm.wide` |
| `experiments/results/` | Every result file used in the paper: JSON outputs of the controlled experiments, M2 plans and results, M3 cached plans (`plans_*.jsonl.gz`), per-problem re-evaluations (`reeval.jsonl`), training logs (`state_*.json`, `job*.log`); `m3w/` holds the multi-part experiment |
| `notebooks/` | Jupyter notebooks for M2 (Google Colab) and M3 (Colab or a RunPod GPU pod) |
| `scripts/cloud/` | Unattended M3 jobs with a watchdog that restarts a stalled run from its last saved step (`job.py`: flat, chain, poset; `job2.py`: random-order chain credit, plus an optional local-credit run for follow-up work; `job3.py`: synthetic wide plans; `reeval.py`: 16-sample re-evaluation) |
| `analysis/` | Scripts that rebuild every figure, table and number in the paper from `experiments/results/`; output goes to `analysis/output/` |
| `tests/` | 114 automated tests: exact routines against brute force, and the full language-model pipeline against a scripted model (no GPU needed) |

## Installation

Python 3.10 or later.

```bash
pip install -e .                 # core: numpy
pip install -e ".[test,plots]"   # tests and figures
pip install -e ".[transformer]"  # JAX experiment
pip install -e ".[llm]"          # language-model experiments (needs a CUDA GPU)
```

## Reproducing the paper

All figures, tables and numbers in the paper are regenerated from the stored results:

```bash
python -m pytest -q tests                  # 114 passed
python analysis/make_figs.py               # controlled experiments, transformer and M2: figures, tables, numbers
python analysis/m3_analysis.py             # M3 on MATH (Table 4, Figure 8)
python analysis/m3w_analysis.py            # M3 on multi-part problems (Table 5, Figure 9)
python experiments/m3_plan_stats.py        # incomparable pairs of the M3 plans
```

The figures (PDF) and the tables and numbers (as LaTeX snippets) are written to `analysis/output/`.

To rerun the experiments themselves:

| Experiment | Command | Hardware, time |
| --- | --- | --- |
| E1, E2, E4 (structure, signal, attenuation) | `python experiments/sim.py e1 e2 e4` | CPU, seconds |
| E3 and noise (learning, verifier noise) | `python experiments/sim.py e3 noise` | CPU, about 2 minutes |
| Shared transformer | `python experiments/transformer_p6.py` | 2 CPU cores, about 40 minutes |
| M2 (frozen model, inference) | `notebooks/M2_inference.ipynb` | one 16 GB GPU (Colab T4), several hours |
| M3 (executor RL, flat/chain/poset) | `scripts/cloud/supervise.sh` (runs `job.py`) then `reeval.py` | one 80 GB GPU, about 7 hours per credit rule |
| M3 (random-order chain) | `scripts/cloud/supervise2.sh` (runs `job2.py`) | one 80 GB GPU, about 7-10 hours per rule |
| M3 on multi-part problems (poset vs random-order chain) | `scripts/cloud/supervise3.sh` (runs `job3.py`; set `STEPS` to 100 and the re-evaluation steps to 25, 50, 75, 100 as in the paper) | one H200 or 80 GB GPU, about 2 hours per credit rule |

The cloud scripts expect the repository at `/workspace/pohrl/code/pohrl` and write to `/workspace/pohrl`; edit
`BASE` at the top of each job file for other locations. They are resumable: a run that stops continues from its last
saved step. Start them with `nohup bash scripts/cloud/supervise.sh > supervise.log 2>&1 &`.

## Data

**Synthetic data.** All synthetic data are generated by scripts in this repository from fixed seeds, so they can be
recreated exactly:

* *Controlled experiments* (E1-E4): random layered plans are drawn by `pohrl.layered_random` inside
  `experiments/sim.py`; no data files are needed.
* *Shared transformer*: plan families (chain, width 3, 5, 15) and arithmetic facts are generated inside
  `experiments/transformer_p6.py`.
* *Multi-part problems* (M3 on plans with many independent subgoals, run on RunPod): `pohrl_llm/wide.py` builds each
  problem from nine templates, with its gold plan and sub-answers. `scripts/cloud/job3.py` calls it with seed 11
  (1,200 training problems, 3-6 parts) and seed 12 (240 evaluation problems, 1-6 parts). The same files are stored in
  `experiments/data/multipart/` and can be regenerated with `python -m pohrl_llm.wide`; a test checks that the
  stored files equal the generator's output.

**Public data.** The language-model experiments on real problems use:

* **MATH** (Hendrycks et al., 2021): <https://huggingface.co/datasets/EleutherAI/hendrycks_math> (original: <https://github.com/hendrycks/math>); 1,200 training problems of levels 3-5.
* **MATH500** (Lightman et al., 2024): <https://huggingface.co/datasets/HuggingFaceH4/MATH-500>; problems of levels 3-5 for evaluation, the 100 hardest in M2.
* **AIME 2024**: <https://huggingface.co/datasets/HuggingFaceH4/aime_2024>; 30 problems, M2.
* **Model**: Qwen3-1.7B, <https://huggingface.co/Qwen/Qwen3-1.7B>.

The data sets are downloaded automatically by the `datasets` library; their licences are those of the original
releases, so the repository stores no problem texts.

**Which problems were used.** `experiments/llm_data.py` repeats the selections made by the experiments (fixed seeds;
see `scripts/cloud/job.py` and `notebooks/M2_inference.ipynb`) and writes a manifest to `experiments/data/llm/`:

| File | Problems |
| --- | --- |
| `m3_train.jsonl` | 1,200 M3 training problems (`tr0` ... `tr1199`): MATH train, levels 3-5, shuffled with seed 0 |
| `m3_eval.jsonl` | 367 MATH500 problems of levels 3-5 (M3 evaluation pool) |
| `m2.jsonl` | 130 M2 problems: the 100 hardest MATH500 problems and all 30 AIME 2024 problems |
| `meta.json` | Hub revisions of the three data sets, the selection rules, and the checks below |

Each line gives the problem's id in the result files, its source data set, configuration, split and row index, its
level, and SHA-256 checksums of the problem text and answer. When it writes the manifest, the script checks the ids
against the stored results, and checks the M3 training problems against the cached plans, whose reasoning begins by restating the problem: among
all 7,500 MATH training problems, the assigned problem is the closest text for 70% of the plans and in the top 20
for 91% (a random pairing gives 0.1% and 0.3%). The manifest in this repository was produced this way; its checks
are recorded in `meta.json`. To build or verify it (internet access, no GPU):

```bash
pip install -e ".[data]"
python experiments/llm_data.py           # write the manifest
python experiments/llm_data.py --check   # fetch the listed problems and verify them against the checksums
```

The function `load_problems` in `experiments/llm_data.py` returns the problems listed in a manifest file.

## Citation

If you use this software, please cite it (see also `CITATION.cff`):

```bibtex
@software{pohrl_software_2026,
  title   = {{PO-HRL}: Partial-Order Hierarchical Reinforcement Learning},
  author  = {Arnauld Mesinga Mwafise},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/GITHUB-USER/po-hrl}
}
```

A citation for the accompanying paper will be added here once it is publicly available.

## Licence

MIT; see `LICENSE`.
