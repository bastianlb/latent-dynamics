#!/usr/bin/env python
"""2e: how much of H1 lifetime is 'noise' (many short bars) and does top-k H1 still mirror Fisher?

Uses the full H1 diagrams cached by compute_extra.py (normalised latents; mean pairwise distance = 1).
Per snapshot: n_bars, total L1, share of L1 carried by bars in the bottom 90% (by persistence rank),
share carried by bars with lifetime < {0.01,0.02,0.05}, and top-k lifetime sums (k=5,10,20).
Outputs: h1_noise_final.csv, fig_h1_noise.png, issue2e_summary.txt
"""
from __future__ import annotations
import common as C
from stats_util import partial_corr

import csv, os
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lines = []
KS = (5, 10, 20)
THR = (0.01, 0.02, 0.05)


def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)


def per_snapshot(ca):
    bars, off = ca["h1_bars"], ca["h1_off"]; out = []
    for i in range(len(off) - 1):
        l = np.sort((bars[off[i]:off[i + 1], 1] - bars[off[i]:off[i + 1], 0]).astype(float))[::-1]
        tot = l.sum() if len(l) else 0.0; n = len(l)
        d = dict(n_bars=n, L1=tot, max_bar=l[0] if n else 0.0)
        cut = int(np.ceil(0.1 * n))                      # top 10% of bars by persistence
        d["share_bottom90"] = l[cut:].sum() / tot if tot > 0 else np.nan
        d["L1_bottom90"] = l[cut:].sum()
        for t in THR: d[f"share_lt{t}"] = l[l < t].sum() / tot if tot > 0 else np.nan
        for k in KS: d[f"top{k}"] = l[:k].sum(); d[f"top{k}_share"] = l[:k].sum() / tot if tot > 0 else np.nan
        out.append(d)
    return {k: np.array([o[k] for o in out]) for k in out[0]}


def main():
    runs = C.list_runs(); rows = []; series = {}
    for r in runs:
        ca = C.load_cache(r["run_id"]); ts = C.load_timeseries(r["run_id"]); s = per_snapshot(ca); series[r["run_id"]] = (s, ts)
        d = dict(r); d["fisher"] = ts["fisher_ratio"][-1]; d["H1_L1_csv"] = ts["h1_lifetime_sum_alpha1.0"][-1]
        d.update({k: v[-1] for k, v in s.items()}); rows.append(d)
    with open(os.path.join(C.OUT, "h1_noise_final.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k not in ("config",)], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    P("## sanity: cache L1 vs csv L1 max rel err =", f"{max(abs(d['L1']-d['H1_L1_csv'])/d['H1_L1_csv'] for d in rows):.2e}")
    P("\n## Final-step H1 composition (mean over runs of that dataset/config group)")
    for ds in C.DATASETS:
        for grp, sel in (("capacity D=2", lambda d: d["objective"] == "probe" and d["norm"] == "none" and d["D"] == 2),
                         ("capacity D=16", lambda d: d["objective"] == "probe" and d["norm"] == "none" and d["D"] == 16),
                         ("l2 D=16", lambda d: d["norm"] == "l2"), ("AE D=16", lambda d: d["objective"] == "autoencoder")):
            rr = [d for d in rows if d["dataset"] == ds and sel(d)]
            P(f"  {ds:13s} {grp:14s} n_bars={np.mean([d['n_bars'] for d in rr]):6.0f}  L1={np.mean([d['L1'] for d in rr]):6.2f}  share from bottom-90% bars={np.mean([d['share_bottom90'] for d in rr]):.2f}  "
              f"share from bars<0.01/0.02/0.05: {np.mean([d['share_lt0.01'] for d in rr]):.2f}/{np.mean([d['share_lt0.02'] for d in rr]):.2f}/{np.mean([d['share_lt0.05'] for d in rr]):.2f}  "
              f"top5/10/20 share={np.mean([d['top5_share'] for d in rr]):.2f}/{np.mean([d['top10_share'] for d in rr]):.2f}/{np.mean([d['top20_share'] for d in rr]):.2f}")
    P("\n## Final-step Spearman with Fisher across runs (n=27; excl AE n=24); partial = controlling log raw-norm & D (excl AE)")
    names = ["L1", "L1_bottom90", "top5", "top10", "top20", "n_bars", "max_bar"]
    for ds in C.DATASETS:
        rr = [d for d in rows if d["dataset"] == ds]; na = [d for d in rr if d["objective"] != "autoencoder"]
        F = np.array([d["fisher"] for d in rr]); Fn = np.array([d["fisher"] for d in na])
        Z = np.column_stack([np.log([C.load_timeseries(d["run_id"])["raw_mean_norm"][-1] for d in na]), [d["D"] for d in na]])
        txt = []
        for n in names:
            x = np.array([d[n] for d in rr]); xn = np.array([d[n] for d in na])
            txt.append(f"{n}:{stats.spearmanr(F,x)[0]:+.2f}/{stats.spearmanr(Fn,xn)[0]:+.2f}/part{partial_corr(Fn,xn,Z):+.2f}")
        P(f"  {ds}: " + "  ".join(txt))
    P("\n## Within-run first-differenced Spearman(dFisher, d X), median over non-AE runs, late window (step>=10%T)")
    for ds in C.DATASETS:
        txt = []
        for n in ["L1", "L1_bottom90", "top5", "top10", "top20", "n_bars"]:
            v = []
            for r in runs:
                if r["dataset"] != ds or r["objective"] == "autoencoder": continue
                s, ts = series[r["run_id"]]; sel = ts["step"][1:] >= 0.1 * ts["step"][-1]
                v.append(stats.spearmanr(np.diff(ts["fisher_ratio"])[sel], np.diff(s[n])[sel])[0])
            txt.append(f"{n}:{np.nanmedian(v):+.2f}")
        P(f"  {ds}: " + "  ".join(txt))
    # figure: share of L1 in bottom 90% vs time, and top-k vs Fisher scatter (final)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ci, ds in enumerate(C.DATASETS):
        ax = axs[0, ci]
        for r in runs:
            if r["dataset"] == ds and r["D"] == 16 and r["seed"] == 0:
                s, ts = series[r["run_id"]]; ax.plot(ts["step"], s["share_bottom90"], label=r["config"])
        ax.set_title(f"{ds}: share of H1 L1 from bottom-90% bars (seed0, D=16)"); ax.set_xscale("symlog", linthresh=50)
        if ci == 0: ax.legend(fontsize=6)
        ax = axs[1, ci]
        for k, c in ((5, "C0"), (20, "C1")):
            ax.scatter([d["fisher"] for d in rows if d["dataset"] == ds], [d[f"top{k}"] for d in rows if d["dataset"] == ds], s=14, c=c, label=f"top{k} sum")
        ax.scatter([d["fisher"] for d in rows if d["dataset"] == ds], [d["L1"] for d in rows if d["dataset"] == ds], s=14, c="0.5", label="all bars L1")
        ax.set_xlabel("final Fisher"); ax.set_xscale("log"); ax.legend(fontsize=7); ax.set_title(f"{ds}: H1 variants vs Fisher (27 runs)")
    fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "fig_h1_noise.png"), dpi=100)
    open(os.path.join(C.OUT, "issue2e_summary.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
