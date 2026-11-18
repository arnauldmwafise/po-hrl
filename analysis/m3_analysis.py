"""M3 analysis: executor RL with flat / chain / poset credit (Qwen3-1.7B, LoRA, one seed).

Reads results/m3/reeval.jsonl (checkpoints at steps 100, 125, 150 and the base model, each
re-evaluated with 16 samples per problem) and writes numbers_m3.tex, tab_m3.tex, figs/fig_m3.pdf.
Accuracy per method = per-problem mean over the three late checkpoints. Intervals: paired
bootstrap over problems (10,000 resamples, seed 0).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"; (OUT / "figs").mkdir(parents=True, exist_ok=True)  # generated figures and LaTeX tables
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else next(p for p in [HERE.parent / "experiments" / "results" / "m3", HERE.parent / "pohrl" / "experiments" / "results" / "m3"] if p.exists())
C = {"base": "#555555", "flat": "#9a9a9a", "chain": "#d9822b", "chain_rand": "#8c4fb0", "poset": "#2a6fdb"}
LBL = {"base": "base model", "flat": "flat (GRPO)", "chain": "chain credit", "chain_rand": "chain, random order", "poset": "poset credit"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titlesize": 9})
METHODS, STEPS, B = ["flat", "chain", "chain_rand", "poset"], (100, 125, 150), 10000

R = {}
for line in open(RES / "reeval.jsonl"):
    r = json.loads(line); R[r["tag"]] = r["rows"]
pids = [x["pid"] for x in R["base@0"]]
assert all([x["pid"] for x in rows] == pids for rows in R.values())
W = np.array([x["width"] for x in R["base@0"]])
SETS = {"all": np.arange(len(W)), "w1": np.where(W == 1)[0], "w23": np.where((W >= 2) & (W <= 3))[0],
        "w4": np.where(W >= 4)[0], "wide": np.where(W >= 2)[0]}
col = lambda tag, k: np.array([x[k] for x in R[tag]])
ACC = {"base": col("base@0", "acc")}
WORDS = {"base": col("base@0", "words_out")}
for m in METHODS:
    ACC[m] = np.mean([col(f"{m}@{s}", "acc") for s in STEPS], 0)
    WORDS[m] = np.mean([col(f"{m}@{s}", "words_out") for s in STEPS], 0)

rng = np.random.default_rng(0)
BOOT = {k: rng.integers(0, len(v), (B, len(v))) for k, v in SETS.items()}


def ci(vec, key):
    v = vec[SETS[key]]; bs = v[BOOT[key]].mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ratio_ci(a, b):
    idx = BOOT["all"]; bs = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


N = {}
p = lambda x: f"{100 * x:.1f}"
pp = lambda x: f"${100 * x:+.1f}$"
names = {"all": "All", "w1": "One", "w23": "TwoThree", "w4": "Four", "wide": "Wide"}
mname = {"base": "Base", "flat": "Flat", "chain": "Chain", "chain_rand": "Rand", "poset": "Poset"}
for m in ["base"] + METHODS:
    for k in SETS:
        N[f"MThree{mname[m]}{names[k]}"] = p(ACC[m][SETS[k]].mean())
for a, b in [("poset", "chain"), ("poset", "chain_rand"), ("chain_rand", "chain"), ("chain_rand", "flat"), ("chain_rand", "base"), ("poset", "flat"), ("chain", "flat"), ("poset", "base"), ("chain", "base"),
             ("flat", "base")]:
    for k in SETS:
        d, lo, hi = ci(ACC[a] - ACC[b], k)
        tag = f"MThree{mname[a]}Minus{mname[b]}{names[k]}"
        N[tag], N[tag + "Lo"], N[tag + "Hi"] = pp(d), pp(lo), pp(hi)
# does the poset-chain gap grow with width?  (wide minus chains, independent bootstrap of the two subsets)
d = ACC["poset"] - ACC["chain"]
i1, iw = SETS["w1"], SETS["wide"]
bs = d[iw][rng.integers(0, len(iw), (B, len(iw)))].mean(1) - d[i1][rng.integers(0, len(i1), (B, len(i1)))].mean(1)
N["MThreeInter"], N["MThreeInterLo"], N["MThreeInterHi"] = pp(d[iw].mean() - d[i1].mean()), *(pp(x) for x in np.percentile(bs, [2.5, 97.5]))
for a, b in [("poset", "flat"), ("chain", "flat"), ("poset", "chain"), ("chain_rand", "chain"), ("chain_rand", "poset"), ("chain_rand", "flat")]:
    r, lo, hi = ratio_ci(WORDS[a], WORDS[b])
    tag = f"MThreeWords{mname[a]}Over{mname[b]}"
    N[tag], N[tag + "Lo"], N[tag + "Hi"] = f"{r:.2f}", f"{lo:.2f}", f"{hi:.2f}"
for m in ["base"] + METHODS:
    N[f"MThreeWords{mname[m]}"] = f"{WORDS[m].mean():,.0f}".replace(",", "{,}")
N["MThreeNeval"], N["MThreeNone"], N["MThreeNtwothree"], N["MThreeNfour"] = (str(len(SETS[k])) for k in ["all", "w1", "w23", "w4"])
# poset-credit share of words dropped relative to chain_rand at step 150 (trend check)
for m in METHODS:
    N[f"MThreeAcc{mname[m]}AtCL"] = p(col(f"{m}@150", "acc").mean())
# per-checkpoint words, to show the ordering is not uniform across checkpoints
for m in METHODS:
    for s in STEPS:
        N[f"MThreeWords{mname[m]}At{ {100: 'C', 125: 'CXXV', 150: 'CL'}[s] }".replace(" ", "")] = f"{col(f'{m}@{s}', 'words_out').mean():,.0f}".replace(",", "{,}")
# incomparable predecessors under the written order (computed from the cached plans with pohrl.plan;
# see the paper repository script; train plans: 7,856 steps)
N["MThreeIncShareTrain"], N["MThreeIncMeanTrain"], N["MThreeIncShareEval"] = "15\\%", "0.35", "17\\%"
N["MThreeIncPairs"], N["MThreeIncShareRand"] = "0.35", "19\\%"   # incomparable pairs per step (order-invariant); share under random orders
N["TabIncPairsLo"], N["TabIncPairsHi"] = "1.4", "6.6"           # incomparable pairs per step, tabular families w2..w15
N["MThreeWordsPosetFewerFlat"] = f"{100*(1-WORDS['poset'].mean()/WORDS['flat'].mean()):.0f}\\%"
# training-set plan widths (from the run log of the notebook; fixed at plan time)
N["MThreeTrainChains"], N["MThreeTrainTwoThree"], N["MThreeTrainFour"], N["MThreeTrainFallback"] = "489", "618", "60", "2.8\\%"

with open(OUT / "numbers_m3.tex", "w") as f:
    f.write("% generated by m3_analysis.py\n")
    for k, v in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def cell(m, k):
    v, lo, hi = ci(ACC[m], k)
    return f"{100*v:.1f} {{\\scriptsize[{100*lo:.0f}, {100*hi:.0f}]}}"


def dcell(a, b, k):
    v, lo, hi = ci(ACC[a] - ACC[b], k)
    return f"${100*v:+.1f}$ {{\\scriptsize$[{100*lo:+.1f}, {100*hi:+.1f}]$}}"


rows = []
for m in ["base"] + METHODS:
    rows.append(f"{LBL[m]} & " + " & ".join(cell(m, k) for k in ["all", "w1", "w23", "w4"]) + f" & {WORDS[m].mean():,.0f}".replace(",", "{,}") + r"\\")
rows.append(r"\midrule")
for a, b in [("poset", "chain"), ("poset", "chain_rand"), ("chain_rand", "chain"), ("poset", "flat")]:
    rows.append(f"{LBL[a].split(' credit')[0]} $-$ {LBL[b].split(' credit')[0]} & " + " & ".join(dcell(a, b, k) for k in ["all", "w1", "w23", "w4"]) + r" & \\")
n = {k: len(v) for k, v in SETS.items()}
tab = r"""\begin{table}[!htbp]
\centering\small
\caption{M3 (Qwen3-1.7B, LoRA, one seed): accuracy (\%%) on MATH500 levels 3--5 by plan-width bucket, averaged over the checkpoints at steps 100, 125 and 150, each evaluated with 16 samples per problem; brackets: 95\%% paired bootstrap intervals over problems. The last column is the mean number of words the executor writes per episode. Lower block: paired differences in points.}\label{tab:m3}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%%
\begin{tabular}{lccccc}
\toprule
 & all ($n=%d$) & width 1 ($%d$) & width 2--3 ($%d$) & width $\ge4$ ($%d$) & words\\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % (n["all"], n["w1"], n["w23"], n["w4"], "\n".join(rows))
(OUT / "tab_m3.tex").write_text(tab)

# figure: (a) accuracy by width bucket with CIs; (b) accuracy vs executor words for every checkpoint
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
keys, labels = ["all", "w1", "w23", "w4"], ["all", "width 1\n(control)", "width 2–3", "width ≥ 4"]
for j, m in enumerate(["base"] + METHODS):
    xs = np.arange(len(keys)) + (j - 2) * 0.16
    st = [ci(ACC[m], k) for k in keys]
    ax[0].bar(xs, [100 * s[0] for s in st], 0.15, color=C[m], label=LBL[m])
    ax[0].errorbar(xs, [100 * s[0] for s in st], yerr=[[100 * (s[0] - s[1]) for s in st], [100 * (s[2] - s[0]) for s in st]],
                   fmt="none", ecolor="k", lw=0.7, capsize=1.5)
ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels(labels); ax[0].set_ylabel("accuracy (%)")
ax[0].legend(frameon=False, ncol=3, fontsize=7, loc="upper center"); ax[0].set_ylim(0, 100)
ax[0].set_title("(a) accuracy by plan width (mean of steps 100–150)")
ax[1].scatter([WORDS["base"].mean()], [100 * ACC["base"].mean()], color=C["base"], marker="*", s=90, label=LBL["base"], zorder=3)
for m in METHODS:
    w = [col(f"{m}@{s}", "words_out").mean() for s in STEPS]
    a = [100 * col(f"{m}@{s}", "acc").mean() for s in STEPS]
    ax[1].plot(w, a, "o-", color=C[m], label=LBL[m], lw=1)
    for s, x, y in zip(STEPS, w, a):
        ax[1].annotate(str(s), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=C[m])
ax[1].set_xlabel("executor words per episode"); ax[1].set_ylabel("accuracy (%)")
ax[1].set_title("(b) accuracy versus output length, per checkpoint"); ax[1].legend(frameon=False, fontsize=7.5)
plt.tight_layout(); plt.savefig(OUT / "figs" / "fig_m3.pdf"); plt.savefig(OUT / "figs" / "fig_m3.png", dpi=150)

for k in ["MThreePosetMinusChainAll", "MThreePosetMinusChainWide", "MThreeInter", "MThreeWordsPosetOverFlat",
          "MThreeWordsPosetOverChain"]:
    print(k, N[k], N.get(k + "Lo"), N.get(k + "Hi"))
print(tab)
