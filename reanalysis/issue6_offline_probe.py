#!/usr/bin/env python
"""Offline linear separability vs the online probe accuracy (esp. autoencoder runs).

offline_probe_acc (cache) = 5-fold CV logistic-regression accuracy on the 1000 probe latents at every snapshot;
final_offline_probe_full.csv (optional) = LR fit on full-train latents of model_final.pt, tested on the full test set.
Outputs: offline_vs_online_final.csv, offline_vs_online_traj.csv, fig_offline_probe.png, issue6_summary.txt
"""
from __future__ import annotations
import common as C

import csv, os
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lines = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)


def main():
    full = {}
    fp = os.path.join(C.OUT, "final_offline_probe_full.csv")
    if os.path.exists(fp): full = {r["run_id"]: float(r["offline_full_test_acc"]) for r in csv.DictReader(open(fp))}
    runs = C.list_runs(); rows = []; traj = []
    for r in runs:
        ts = C.load_timeseries(r["run_id"]); ca = C.load_cache(r["run_id"])
        on, off = ts["test_acc"], ca["offline_probe_acc"]
        rows.append(dict(run_id=r["run_id"], dataset=r["dataset"], config=r["config"], seed=r["seed"], online_test_acc=on[-1],
                         offline_cv_acc=off[-1], offline_full_test_acc=full.get(r["run_id"], np.nan)))
        traj.append(dict(run_id=r["run_id"], dataset=r["dataset"], config=r["config"], rho_time=stats.spearmanr(on, off)[0],
                         mean_diff=np.mean(on - off), mean_abs_diff=np.mean(np.abs(on - off)), diff_first10=np.mean((on - off)[:10]), diff_last10=np.mean((on - off)[-10:]),
                         steps_to_90pct_online=ts["step"][np.argmax(on >= 0.9 * on[-1])], steps_to_90pct_offline=ts["step"][np.argmax(off >= 0.9 * off[-1])]))
    for fn, rr in (("offline_vs_online_final.csv", rows), ("offline_vs_online_traj.csv", traj)):
        with open(os.path.join(C.OUT, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    P("## Final snapshot: online test_acc | offline CV (1000 probe pts) | offline full-data probe (model_final); mean over 3 seeds")
    cfgs = ["probe/none/d16", "probe_recon/none/d16", "probe/l2/d16", "probe/bn/d16", "autoencoder/none/d16"]
    for ds in C.DATASETS:
        for c in cfgs:
            rr = [d for d in rows if d["dataset"] == ds and d["config"] == c]
            P(f"  {ds:13s} {c:22s} online={np.mean([d['online_test_acc'] for d in rr]):.3f}  offlineCV={np.mean([d['offline_cv_acc'] for d in rr]):.3f}  offlineFull={np.nanmean([d['offline_full_test_acc'] for d in rr]):.3f}")
    P("\n## Over time (all snapshots): per-config mean of online - offlineCV (first 10 snapshots / last 10), median within-run Spearman(online,offline)")
    for ds in C.DATASETS:
        for c in cfgs:
            rr = [d for d in traj if d["dataset"] == ds and d["config"] == c]
            P(f"  {ds:13s} {c:22s} diff_first10={np.mean([d['diff_first10'] for d in rr]):+.3f}  diff_last10={np.mean([d['diff_last10'] for d in rr]):+.3f}  rho={np.median([d['rho_time'] for d in rr]):+.2f}  "
              f"t90 online/offline steps={np.mean([d['steps_to_90pct_online'] for d in rr]):.0f}/{np.mean([d['steps_to_90pct_offline'] for d in rr]):.0f}")
    P("\n## Calibration of offlineCV vs offlineFull across all runs:")
    ok = [d for d in rows if np.isfinite(d["offline_full_test_acc"])]
    if ok:
        a = np.array([d["offline_cv_acc"] for d in ok]); b = np.array([d["offline_full_test_acc"] for d in ok]); o = np.array([d["online_test_acc"] for d in ok])
        P(f"  n={len(ok)}  mean(offlineCV - offlineFull)={np.mean(a-b):+.4f} (sd {np.std(a-b):.4f}); mean(online - offlineFull)={np.mean(o-b):+.4f} (sd {np.std(o-b):.4f})")
        ae = np.array([d["config"].startswith("autoencoder") for d in ok])
        P(f"  AE only: mean(online - offlineFull)={np.mean((o-b)[ae]):+.4f}; non-AE: {np.mean((o-b)[~ae]):+.4f}")
        P(f"  AE only: mean(offlineCV - offlineFull)={np.mean((a-b)[ae]):+.4f}; non-AE: {np.mean((a-b)[~ae]):+.4f}")
    # does offline probe acc add info to Fisher? rank corr across runs
    P("\n## Spearman across 27 runs: offlineCV vs Fisher / H1_L1 / PMag (final step)")
    for ds in C.DATASETS:
        rr = [r for r in runs if r["dataset"] == ds]
        off = [C.load_cache(r["run_id"])["offline_probe_acc"][-1] for r in rr]
        ts = [C.load_timeseries(r["run_id"]) for r in rr]
        P(f"  {ds}: " + "  ".join(f"{k}:{stats.spearmanr(off,[t[c][-1] for t in ts])[0]:+.2f}" for k, c in (("fisher", "fisher_ratio"), ("H1_L1", "h1_lifetime_sum_alpha1.0"), ("pmag", "pmag"), ("online_acc", "test_acc"))))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, ds in zip(axs, C.DATASETS):
        for c, col in zip(cfgs, ("C0", "C1", "C2", "C4", "C3")):
            for s in (0, 1, 2):
                rid = f"{ds}_{c.replace('/d16','').replace('/','_')}_d16_s{s}"
                ts = C.load_timeseries(rid); off = C.load_cache(rid)["offline_probe_acc"]
                ax.plot(ts["step"], ts["test_acc"], c=col, lw=1.2, label=c if s == 0 else None); ax.plot(ts["step"], off, c=col, ls="--", lw=1)
        ax.set_xscale("symlog", linthresh=50); ax.set_title(f"{ds}: online test_acc (solid) vs offline CV probe (dashed)")
    axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "fig_offline_probe.png"), dpi=100)
    open(os.path.join(C.OUT, "issue6_summary.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
