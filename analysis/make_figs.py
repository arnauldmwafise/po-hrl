"""Build every figure, table and number in the paper from the experiment result files.

Usage: python make_figs.py [results_dir]   (default ../pohrl/experiments/results)
Writes figs/*.pdf, numbers.tex, tab_signal.tex, tab_transformer.tex.
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"; (OUT / "figs").mkdir(parents=True, exist_ok=True)  # generated figures and LaTeX tables
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else next(p for p in [HERE.parent / "experiments" / "results", HERE.parent / "pohrl" / "experiments" / "results"] if p.exists())
FIG = OUT / "figs"
C = {"flat": "#9a9a9a", "ideal_bonus": "#c9b38a", "chain": "#d9822b", "poset": "#2a6fdb", "local": "#2e9e5b"}
LBL = {"flat": "flat GRPO", "ideal_bonus": "ideal bonus", "chain": "chain credit", "poset": "poset credit",
       "local": "local credit"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titlesize": 9})
N = {}


def pct(x, d=0):
    return f"{100 * x:.{d}f}\\%"


def sci(x):
    e = int(np.floor(np.log10(x))); m = x / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


# ------------------------------------------------------------------ E1
e1 = json.loads((RES / "e1_structure.json").read_text())
W = [r["width"] for r in e1]
by = {r["width"]: r for r in e1}
N["EOnePlans"] = str(sum(r["plans"] for r in e1))
N["EOnePrefixWnine"] = sci(by[9]["median_prefix_nodes_b3"])
N["EOneIdealWnine"] = sci(by[9]["median_ideal_dag_nodes_b3"])
N["EOneRoundsWone"] = f"{by[1]['mean_rounds_height']:.0f}"
N["EOneRoundsWnine"] = f"{by[9]['mean_rounds_height']:.0f}"
N["EOneCtxWnine"] = pct(by[9]["mean_context_ratio"])
keep = [by[w]["mean_kept_after_failure_chain"] for w in W if w >= 2]
N["EOneKeepRange"] = f"{100 * min(keep):.0f}--{100 * max(keep):.0f}\\%"
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
ax[0].semilogy(W, [by[w]["median_prefix_nodes_b3"] for w in W], "o-", color=C["chain"], label="prefix tree (orderings)")
ax[0].semilogy(W, [by[w]["median_ideal_dag_nodes_b3"] for w in W], "o-", color=C["poset"], label="ideal-keyed")
ax[0].set_xlabel("plan width"); ax[0].set_ylabel("search nodes ($b=3$)"); ax[0].legend(frameon=False)
ax[1].plot(W, [by[w]["mean_rounds_height"] for w in W], "o-", color=C["poset"], label="parallel rounds")
ax[1].set_xlabel("plan width"); ax[1].set_ylabel("rounds")
a2 = ax[1].twinx(); a2.plot(W, [by[w]["mean_context_ratio"] for w in W], "s--", color="#555", label="context ratio")
a2.set_ylabel("principal-ideal / full context"); a2.set_ylim(0, 1.05); a2.spines["top"].set_visible(False)
ax[1].legend(loc="upper right", frameon=False); a2.legend(loc="center right", frameon=False)
ax[2].plot(W, [by[w]["mean_kept_after_failure_poset"] for w in W], "o-", color=C["poset"], label="filter repair")
ax[2].plot(W, [by[w]["mean_kept_after_failure_chain"] for w in W], "o-", color=C["chain"], label="chain repair")
ax[2].set_ylim(0, 1.05); ax[2].set_xlabel("plan width"); ax[2].set_ylabel("work kept after a failure")
ax[2].legend(frameon=False)
plt.tight_layout(); plt.savefig(FIG / "fig_structure.pdf"); plt.close()

# ------------------------------------------------------------------ E2
e2 = json.loads((RES / "e2_signal.json").read_text())
N["ETwoOutcomeMin"] = pct(min(r["outcome"] for r in e2))
N["ETwoOutcomeMax"] = pct(max(r["outcome"] for r in e2))
rows = []
fam = {"chain": "chain", "w3": "width 3", "w9": "width 9"}
for r in e2:
    rows.append(f"{fam[r['family']]} & {r['p']:.1f} & {pct(r['outcome'], 1)} & {pct(r['chain_prefix'], 2)} & {pct(r['ideal'], 2)} \\\\")
(OUT / "tab_signal.tex").write_text(r"""\begin{table}[!htbp]
\centering\small
\caption{Share of groups ($G=8$) whose reward carries no signal (all rewards equal), for 10-subgoal plans and per-subgoal success $p$; 4{,}000 groups per cell.}\label{tab:signal}
\begin{tabular}{llccc}
\toprule
plan & $p$ & outcome only & chain prefix & verified ideal\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ E4
e4 = json.loads((RES / "e4_attenuation.json").read_text())
N["EFourPairs"] = f"{e4['pairs']:,}".replace(",", "{,}")
N["EFourMAE"] = f"{e4['mean_abs_error']:.3f}"
N["EFourPred"] = f"{e4['mean_predicted_ratio']:.3f}"
N["EFourMeas"] = f"{e4['mean_measured_ratio']:.3f}"
N["EFourLossPct"] = pct(1 - e4["mean_measured_ratio"])
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].scatter(e4["pairs_predicted"], e4["pairs_measured"], s=4, alpha=0.3, color=C["poset"])
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xlabel("predicted $A_g(L)$ (Theorem 1)")
ax[0].set_ylabel("measured chain / poset credit")
ks = [a["k"] for a in e4["antichain"]]
ax[1].plot(ks, [a["closed_form"] for a in e4["antichain"]], "k-", label="closed form (Cor. 2)")
ax[1].plot(ks, [a["simulated"] for a in e4["antichain"]], "o", color=C["chain"], label="simulation")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("antichain size $k$ ($p=0.75$)")
ax[1].set_ylabel("mean attenuation"); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(FIG / "fig_attenuation.pdf"); plt.close()

# ------------------------------------------------------------------ E3
e3 = json.loads((RES / "e3_learning.json").read_text())
S = {(r["regime"], r["family"], r["method"]): r for r in e3["summary"]}
wmed = {"chain": 1, "w2": 2, "w3": 4.5, "w5": 7, "w8": 9, "w15": 15}
fams = ["chain", "w2", "w3", "w5", "w8", "w15"]
N["EThreeSeeds"] = str(len(S[("hard", "chain", "poset")]["per_seed_iters_to_50pct"]))
ci = [S[("hard", f, "chain")]["median_iters_to_50pct"] for f in fams]
N["EThreeChainIters"] = f"{min(ci):.0f}--{max(ci):.0f}"
N["EThreePosetWone"] = f"{S[('hard', 'chain', 'poset')]['median_iters_to_50pct']:.0f}"
N["EThreePosetWfifteen"] = f"{S[('hard', 'w15', 'poset')]['median_iters_to_50pct']:.0f}"
N["EThreeSpeedup"] = f"{S[('hard', 'w15', 'chain')]['median_iters_to_50pct'] / S[('hard', 'w15', 'poset')]['median_iters_to_50pct']:.1f}$\\times$"
N["EThreePosetAtFifty"] = pct(S[("hard", "w15", "poset")]["success_at_50"])
N["EThreeChainAtFifty"] = pct(S[("hard", "w15", "chain")]["success_at_50"], 1)
N["EThreeBonusFinal"] = pct(S[("hard", "w15", "ideal_bonus")]["final"])
N["EThreePosetFinal"] = pct(S[("hard", "w15", "poset")]["final"])
easy_c = [S[("easy", f, "chain")]["median_iters_to_50pct"] for f in ["chain", "w2", "w3", "w5", "w9"]]
easy_p = [S[("easy", f, "poset")]["median_iters_to_50pct"] for f in ["chain", "w2", "w3", "w5", "w9"]]
N["EThreeEasyChain"] = f"{np.median(easy_c):.0f}"
N["EThreeEasyPosetRange"] = f"{min(easy_p):.0f}--{max(easy_p):.0f}"
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for m in ["chain", "poset"]:
    med, lo, hi = [], [], []
    for f in fams:
        v = np.array(S[("hard", f, m)]["per_seed_iters_to_50pct"], float)
        med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
    x = [wmed[f] for f in fams]
    ax[0].plot(x, med, "o-", color=C[m], label=LBL[m]); ax[0].fill_between(x, lo, hi, color=C[m], alpha=0.15)
ax[0].set_xlabel("plan width (median)"); ax[0].set_ylabel("iterations to 50% success"); ax[0].set_ylim(0, None)
ax[0].legend(frameon=False)
for m in ["flat", "ideal_bonus", "chain", "poset"]:
    ax[1].plot(e3["hard"][f"w15|{m}"], color=C[m], label=LBL[m])
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("task success (width 15)"); ax[1].legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "fig_learning.pdf"); plt.close()

# ------------------------------------------------------------------ noise
nz = json.loads((RES / "e3_noise.json").read_text())["summary"]
NS = {(r["noise"], r["method"]): r for r in nz}
N["NoiseTop"] = pct(0.3)
for e, tag in [(0.1, "Ten"), (0.2, "Twenty"), (0.3, "Top")]:
    N[f"NoisePoset{tag}"] = pct(NS[(e, "poset")]["final"])
    N[f"NoiseChain{tag}"] = pct(NS[(e, "chain")]["final"])
fig, ax = plt.subplots(figsize=(5, 2.8))
eps = [0.0, 0.1, 0.2, 0.3]
for k, m in enumerate(["chain", "poset"]):
    vals = [NS[(e, m)]["final"] for e in eps]
    se = [np.std(NS[(e, m)]["per_seed_final"], ddof=1) / np.sqrt(len(NS[(e, m)]["per_seed_final"])) for e in eps]
    ax.bar(np.arange(4) + (k - 0.5) * 0.38, vals, 0.36, yerr=se, color=C[m], label=LBL[m], capsize=2)
ax.set_xticks(range(4)); ax.set_xticklabels([pct(e).replace("\\", "") for e in eps])
ax.set_xlabel("share of subgoal verdicts flipped"); ax.set_ylabel("final task success"); ax.set_ylim(0, 1.18)
ax.legend(frameon=False, ncol=2, loc="upper center")
plt.tight_layout(); plt.savefig(FIG / "fig_noise.pdf"); plt.close()

# ------------------------------------------------------------------ transformer
tp = json.loads((RES / "transformer_p6.json").read_text())
tfam = ["chain", "w3", "w5", "w15"]; tname = {"chain": "chain", "w3": "width 3", "w5": "width 5", "w15": "width 15"}
tmeth = [m for m in ["flat", "chain", "poset"] if any(k.split("|")[1] == m for k in tp)]
seeds = sorted({int(k.split("|")[2]) for k in tp})
N["TFSeeds"] = str(len(seeds))
warm = np.mean([tp[f"chain|{tmeth[0]}|{s}"][0]["table_acc"] for s in seeds])
N["TFWarm"] = pct(warm, 1)


def final(f, m, key="table_acc"):
    v = np.array([tp[f"{f}|{m}|{s}"][-1][key] for s in seeds if f"{f}|{m}|{s}" in tp])
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


N["TFPosetWfifteen"] = pct(final("w15", "poset")[0], 1)
N["TFChainWfifteen"] = pct(final("w15", "chain")[0], 1)
lines = []
for f in tfam:
    cells = []
    for m in tmeth:
        mu, se = final(f, m)
        cells.append(f"{100 * mu:.1f} $\\pm$ {100 * se:.1f}")
    lines.append(f"{tname[f]} & " + " & ".join(cells) + " \\\\")
(OUT / "tab_transformer.tex").write_text(r"""\begin{table}[!htbp]
\centering\small
\caption{Shared transformer executor: fact-table accuracy (\%) after 400 RL steps, mean $\pm$ standard error over """ + str(len(seeds)) + r""" seeds. Warm-start accuracy: """ + f"{100 * warm:.1f}" + r"""\%; flat GRPO receives no signal and stays there.}\label{tab:transformer}
\begin{tabular}{l""" + "c" * len(tmeth) + r"""}
\toprule
plan & """ + " & ".join(LBL[m] for m in tmeth) + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
fig, ax = plt.subplots(1, len(tfam), figsize=(11, 2.8), sharey=True)
for i, f in enumerate(tfam):
    for m in tmeth:
        runs = [tp[f"{f}|{m}|{s}"] for s in seeds if f"{f}|{m}|{s}" in tp]
        steps = [r["step"] for r in runs[0]]
        Y = np.array([[r["table_acc"] for r in run] for run in runs])
        mu = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else 0 * mu
        ax[i].plot(steps, mu, color=C[m], label=LBL[m]); ax[i].fill_between(steps, mu - se, mu + se, color=C[m], alpha=0.15)
    ax[i].axhline(warm, color="k", ls="--", lw=0.7)
    ax[i].set_title(tname[f]); ax[i].set_xlabel("RL step")
ax[0].set_ylabel("fact-table accuracy")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h + [plt.Line2D([], [], color="k", ls="--", lw=0.7)], l + ["warm start"], loc="lower center",
           ncol=len(l) + 1, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(FIG / "fig_transformer.pdf"); plt.close()

# ------------------------------------------------------------------ tests
try:
    TESTS = next(p for p in [HERE.parent / "tests", HERE.parent / "pohrl" / "tests"] if p.exists())
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", str(TESTS)],
                         capture_output=True, text=True, cwd=TESTS.parent).stdout
    import re
    N["TestCount"] = re.search(r"(\d+) passed", out).group(1)
except Exception:
    N["TestCount"] = "??"

(OUT / "numbers.tex").write_text("% generated by make_figs.py from the result files; do not edit\n" +
                                  "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(N.items())) + "\n")
print(json.dumps(N, indent=1))

# ------------------------------------------------------------------ M2 (real LLM, frozen Qwen3-1.7B)
import random as _random
import statistics as _S
import collections as _C


def _load_m2(d):
    P = {json.loads(l)["pid"]: json.loads(l) for l in open(d / "m2_plans.jsonl")}
    R = {json.loads(l)["pid"]: json.loads(l) for l in open(d / "m2_results.jsonl")}
    return P, R


def _boot(d, n=4000):
    rng = _random.Random(0)
    m = sorted(_S.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return _S.mean(d), m[int(0.025 * n)], m[int(0.975 * n)]


if (RES / "m2_v2" / "m2_results.jsonl").exists():
    P1, R1 = _load_m2(RES / "m2_v1")
    P2, R2 = _load_m2(RES / "m2_v2")
    ids = list(R2)
    pl = [R2[k] for k in ids if not R2[k]["fallback"]]
    N["MTwoProblems"] = str(len(ids))
    N["MTwoFallbackOne"] = pct(_S.mean(R1[k]["fallback"] for k in ids))
    N["MTwoFallbackTwo"] = pct(_S.mean(R2[k]["fallback"] for k in ids))
    N["MTwoForced"] = str(sum(P2[k].get("forced", False) for k in ids))
    N["MTwoPlanned"] = str(len(pl))
    N["MTwoShareWide"] = pct(_S.mean(r["width"] >= 2 for r in pl))
    aime_pl = [r for r in pl if r["pid"].startswith("aime")]; math_pl = [r for r in pl if not r["pid"].startswith("aime")]
    N["MTwoShareWideAIME"] = pct(_S.mean(r["width"] >= 2 for r in aime_pl))
    N["MTwoShareWideMATH"] = pct(_S.mean(r["width"] >= 2 for r in math_pl))
    N["MTwoShareWideFour"] = pct(_S.mean(r["width"] >= 4 for r in pl))
    N["MTwoMaxLinExt"] = f"{max(r['linear_extensions'] or 0 for r in pl):,}".replace(",", "{,}")
    N["MTwoMeanSteps"] = f"{_S.mean(r['n'] for r in pl):.1f}"
    N["MTwoMeanHeight"] = f"{_S.mean(r['height'] for r in pl):.2f}"
    N["MTwoMeanRounds"] = f"{_S.mean(r['rounds'] for r in pl):.2f}"
    # incomparable pairs per step (Proposition 4), from the stored plans with the released parser
    try:
        sys.path.insert(0, str(RES.parents[1])); sys.path.insert(0, str(RES.parents[1] / "experiments"))
        from pohrl import plan as _planlib
        from m3_plan_stats import iota as _iota
        def _ip(sel):
            k = n = 0
            for r in sel:
                v = _planlib.validate(P2[r["pid"]]["plan"]); assert not v.fallback and len(list(v.poset.elements)) == r["n"]
                k += _iota(v.poset); n += r["n"]
            return k / n
    except Exception as _e:  # pragma: no cover
        print("incomparable pairs unavailable:", _e); _ip = None
    if _ip:
        N["MTwoIncPairs"] = f"{_ip(pl):.2f}"
    rows = []
    def _row(label, sel):
        if not sel:
            return
        cells = [label, str(len(sel)), f"{100 * _S.mean(r['width'] >= 2 for r in sel):.0f}",
                 f"{100 * _S.mean(r['width'] >= 4 for r in sel):.0f}", f"{_S.mean(r['n'] for r in sel):.1f}",
                 f"{_S.mean(r['rounds'] for r in sel):.2f}",
                 (f"{_ip(sel):.2f}" if _ip else "--"),
                 f"{max(r['linear_extensions'] or 0 for r in sel):,}".replace(",", "{,}")]
        rows.append(" & ".join(cells) + " \\\\")
    _row("all problems", pl)
    _row("\\quad MATH500 (100 hardest)", [r for r in pl if not r["pid"].startswith("aime")])
    _row("\\quad AIME 2024", [r for r in pl if r["pid"].startswith("aime")])
    ch = [k for k in ids if not R2[k]["fallback"] and P2[k].get("audited") and P2[k]["pre_audit_edges"] != P2[k]["edges"]]
    N["MTwoAuditChanged"] = str(len(ch))
    (OUT / "tab_m2.tex").write_text(r"""\begin{table}[!htbp]
\centering\small
\caption{M2: structure of the """ + str(len(pl)) + r""" valid plans written by a frozen Qwen3-1.7B for """ + str(len(ids)) + r""" problems. Non-chains: plans of width at least 2. Rounds: mean number of parallel rounds when executing the plan (\cref{alg:episode}). Pairs per step: incomparable pairs $\iota(\Pset)$ divided by the number of steps, aggregated over plans (\cref{prop:pairs}). Linear extensions: largest number in one plan.}\label{tab:m2}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\toprule
 & & non-chains & width $\ge4$ & & & pairs & linear\\
subset & plans & (\%) & (\%) & steps & rounds & per step & extensions\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
    wc = _C.Counter(min(r["width"], 6) for r in pl)
    xs = list(range(1, 7))
    ax[0].bar(xs, [wc.get(x, 0) for x in xs], color=[C["chain"] if x == 1 else C["poset"] for x in xs])
    ax[0].set_xticks(xs); ax[0].set_xticklabels([str(x) for x in xs[:-1]] + ["6+"])
    ax[0].set_xlabel("plan width (largest antichain)"); ax[0].set_ylabel("plans")
    ax[0].set_title(f"{N['MTwoShareWide'].replace(chr(92), '')} of {len(pl)} plans are not chains")
    buckets = [("1", lambda w: w == 1), ("2", lambda w: w == 2), ("3", lambda w: w == 3), ("4+", lambda w: w >= 4)]
    st, rd, labs = [], [], []
    for lab, f in buckets:
        sel = [r for r in pl if f(r["width"])]
        st.append(_S.mean(r["n"] for r in sel)); rd.append(_S.mean(r["rounds"] for r in sel)); labs.append(f"{lab}\n(n={len(sel)})")
    x = np.arange(len(buckets))
    ax[1].bar(x - 0.2, st, 0.4, color=C["chain"], label="steps (sequential)")
    ax[1].bar(x + 0.2, rd, 0.4, color=C["poset"], label="parallel rounds")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs); ax[1].set_xlabel("plan width")
    ax[1].set_ylabel("mean per plan"); ax[1].legend(frameon=False); ax[1].set_title("rounds needed versus steps written")
    plt.tight_layout(); plt.savefig(FIG / "fig_m2.pdf"); plt.close()
    (OUT / "numbers.tex").write_text("% generated by make_figs.py from the result files; do not edit\n" +
                                      "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(N.items())) + "\n")
    print({k: v for k, v in N.items() if k.startswith("MTwo")})
