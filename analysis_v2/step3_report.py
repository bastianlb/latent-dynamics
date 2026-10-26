#!/usr/bin/env python
"""Figures + text summary for step 3 from step3_results.csv."""
import common2 as C
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(f"{C.OUT}/step3_results.csv")))
f = lambda x: float(x) if x not in ("", None) else np.nan
CUTS = sorted({int(r["cutoff"]) for r in rows})
TG = ["gap_15k", "acc_15k", "gap_pre", "acc_pre"]
out = []
P = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))


def get(tag, tgt, model, kind, name, base=""):
    return [next((r for r in rows if r["tag"] == tag and r["target"] == tgt and int(r["cutoff"]) == c and r["model"] == model
                  and r["kind"] == kind and r["set"] == name and r["base"] == base), None) for c in CUTS]


for model in ("ridge", "gbm"):
    fig, axs = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
    for j, t in enumerate(TG):
        for name, st in (("B0", "--"), ("B1", "-"), ("B0+G", "--"), ("B1+G", "-"), ("G", ":")):
            rs = get("main", t, model, "set", name)
            if rs[0] is None: continue
            col = {"B0": "tab:gray", "B1": "k", "B0+G": "tab:orange", "B1+G": "tab:red", "G": "tab:blue"}[name]
            axs[0, j].plot(CUTS, [f(r["r2"]) for r in rs], st, marker="o", c=col, label=name)
        axs[0, j].set_title(t); axs[0, j].set_xscale("log"); axs[0, j].set_ylim(-0.5, 1)
        for base, comp, col in (("B0", "B0+G", "tab:orange"), ("B1", "B1+G", "tab:red")):
            rs = get("main", t, model, "increment", comp, base)
            if rs[0] is None: continue
            d = np.array([f(r["delta"]) for r in rs]); lo = np.array([f(r["ci_lo"]) for r in rs]); hi = np.array([f(r["ci_hi"]) for r in rs])
            axs[1, j].errorbar(np.array(CUTS) * (1.04 if base == "B1" else 1), d, [d - lo, hi - d], marker="o", c=col, label=f"{base} -> {comp}", capsize=3)
        axs[1, j].axhline(0, c="k", lw=.5); axs[1, j].set_xlabel("cutoff step")
    axs[0, 0].set_ylabel(f"LOCO R^2 ({model})"); axs[1, 0].set_ylabel("delta R^2 (95% config bootstrap)")
    axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
    fig.suptitle(f"Early prediction, leave-one-config-out, {model}")
    fig.savefig(f"{C.FIG}/fig3_r2_vs_cutoff_{model}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

for tag in ("main", "sens_excl_slingshot_le15k", "exploratory_noisy_only"):
    for model in ("ridge", "gbm"):
        for t in sorted({r["target"] for r in rows if r["tag"] == tag}):
            P(f"\n[{tag}] target={t} model={model}  (n_runs={next((r['n_runs'] for r in rows if r['tag']==tag and r['target']==t), '')})")
            for name in ("B0", "B1", "G", "B0+G", "B1+G"):
                rs = get(tag, t, model, "set", name)
                if rs[0] is None: continue
                P(f"  R2 {name:6s} " + " ".join(f"{f(r['r2']):+.2f}" for r in rs) + "   (cutoffs " + ",".join(map(str, CUTS)) + ")")
            for base, comp in (("B0", "B0+G"), ("B1", "B1+G")):
                rs = get(tag, t, model, "increment", comp, base)
                if rs[0] is None: continue
                P(f"  dR2 {base}->{comp}: " + " ".join(f"{f(r['delta']):+.2f}[{f(r['ci_lo']):+.2f},{f(r['ci_hi']):+.2f}]" + (f"p={f(r['perm_p']):.3f}" if r['perm_p'] != '' else '') for r in rs))
open(f"{C.OUT}/step3_summary.txt", "w").write("\n".join(out))
