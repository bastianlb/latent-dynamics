#!/usr/bin/env python
"""Convergence timing with baseline-normalised progress  p(t) = (x_t - x_0) / (x_T - x_0).

For test accuracy and each metric, per run: first snapshot step with p >= 0.5 / 0.9
(t50, t90; also as fraction of total steps), and the share of the total change that has
already happened at the first snapshot after step 0 (p at the 2nd snapshot).
Outputs: timing_per_run.csv, timing_summary.csv, fig_timing_*.png
Uses the existing metrics/timeseries.csv (normalised-latent metrics) + online test_acc, and
offline_probe_acc from the compute_extra cache if present.
"""
from __future__ import annotations
import common as C

import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = {"test_acc": "test_acc", "fisher": "fisher_ratio", "h1_L1": "h1_lifetime_sum_alpha1.0",
           "h1_entropy": "h1_entropy", "pmag": "pmag", "raw_norm": "raw_mean_norm"}


def progress(x):
    d = x[-1] - x[0]
    return (x - x[0]) / d if abs(d) > 1e-12 else np.full_like(x, np.nan)


def settle_step(p, steps, tol=0.1):
    """First step after which p stays within +-tol of 1 (settling time)."""
    bad = np.where(~(np.abs(p - 1) <= tol))[0]
    if len(bad) == 0: return steps[0]
    return steps[bad[-1] + 1] if bad[-1] + 1 < len(steps) else np.nan


def first_cross(p, steps, thr):
    idx = np.where(p >= thr)[0]
    return (steps[idx[0]], idx[0]) if len(idx) else (np.nan, -1)


def main():
    os.makedirs(C.OUT, exist_ok=True)
    rows = []
    runs = [r for r in C.list_runs() if r["objective"] == "probe" and r["norm"] == "none"]
    for r in runs:
        ts = C.load_timeseries(r["run_id"]); steps = ts["step"]; T = steps[-1]
        series = {k: ts[v] for k, v in METRICS.items()}
        cache = C.load_cache(r["run_id"])
        if cache is not None:
            series["offline_probe_acc"] = cache["offline_probe_acc"]
        for k, x in series.items():
            p = progress(np.asarray(x, float))
            t50, i50 = first_cross(p, steps, .5); t90, i90 = first_cross(p, steps, .9)
            rows.append(dict(dataset=r["dataset"], D=r["D"], seed=r["seed"], metric=k,
                             x0=x[0], xT=x[-1], p_first=p[1], t50=t50, t90=t90,
                             f50=t50 / T, f90=t90 / T, settle90=settle_step(p, steps) / T, censored90=int(i90 <= 1), censored50=int(i50 <= 1),
                             naive90=(first_cross(np.asarray(x, float) / x[-1], steps, .9)[0] / T) if (x[-1] > x[0] > -1e9 and k in ("test_acc", "offline_probe_acc", "fisher", "pmag")) else np.nan,
                             n_snap=len(steps), T=T))
    path = os.path.join(C.OUT, "timing_per_run.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # summary across seeds
    summ = []
    keys = sorted({(r["dataset"], r["D"], r["metric"]) for r in rows})
    for ds, D, m in keys:
        rr = [r for r in rows if (r["dataset"], r["D"], r["metric"]) == (ds, D, m)]
        g = lambda k: np.array([r[k] for r in rr], float)
        summ.append(dict(dataset=ds, D=D, metric=m, n=len(rr),
                         f50_mean=np.nanmean(g("f50")), f50_sd=np.nanstd(g("f50")),
                         f90_mean=np.nanmean(g("f90")), f90_sd=np.nanstd(g("f90")),
                         p_first_mean=np.nanmean(g("p_first")), p_first_sd=np.nanstd(g("p_first")),
                         settle90_mean=np.nanmean(g("settle90")), settle90_sd=np.nanstd(g("settle90")),
                         n_censored90=int(g("censored90").sum()), n_censored50=int(g("censored50").sum()),
                         naive90_mean=np.nanmean(g("naive90")) if np.isfinite(g("naive90")).any() else np.nan,
                         x0=np.mean(g("x0")), xT=np.mean(g("xT"))))
    with open(os.path.join(C.OUT, "timing_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    order = ["test_acc", "offline_probe_acc", "fisher", "h1_L1", "h1_entropy", "pmag", "raw_norm"]
    print("t90 as % of training (mean over 3 seeds), p_first = share of total change at 1st snapshot after 0")
    for ds in C.DATASETS:
        print(f"== {ds}  D=16 probe/none")
        for m in order:
            s = [x for x in summ if x["dataset"] == ds and x["D"] == 16 and x["metric"] == m]
            if s:
                s = s[0]
                print(f"  {m:18s} t50={100*s['f50_mean']:5.1f}%±{100*s['f50_sd']:.1f}  t90={100*s['f90_mean']:5.1f}%±{100*s['f90_sd']:.1f} "
                      f"settle(+-10%)={100*s['settle90_mean']:5.1f}%  naive90(x_t>=.9x_T)={100*s['naive90_mean']:5.1f}% p_first={s['p_first_mean']:+.2f}±{s['p_first_sd']:.2f} cens90={s['n_censored90']}/3 x0={s['x0']:.3g} xT={s['xT']:.3g}")
    print("\n== capacity sweep: t90 % (acc | fisher | h1_L1 | pmag) and p_first(acc|fisher)")
    for ds in C.DATASETS:
        for D in (2, 4, 8, 16, 32):
            g = lambda m, k: [x for x in summ if x["dataset"] == ds and x["D"] == D and x["metric"] == m][0][k]
            print(f"  {ds:14s} D={D:2d} t90: " + " | ".join(f"{100*g(m,'f90_mean'):5.1f}" for m in ("test_acc", "fisher", "h1_L1", "pmag")) +
                  f"   p_first: {g('test_acc','p_first_mean'):.2f} {g('fisher','p_first_mean'):.2f} {g('h1_L1','p_first_mean'):.2f} {g('pmag','p_first_mean'):.2f}")

    # figure: progress curves, D=16, mean over seeds
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    cols = dict(zip(order, plt.cm.tab10.colors))
    for ax, ds in zip(axs, C.DATASETS):
        for m in order:
            ps = []
            for r in runs:
                if r["dataset"] == ds and r["D"] == 16:
                    ts = C.load_timeseries(r["run_id"])
                    if m == "offline_probe_acc":
                        c = C.load_cache(r["run_id"])
                        if c is None: continue
                        x = c["offline_probe_acc"]
                    else:
                        x = ts[METRICS[m]]
                    ps.append(progress(np.asarray(x, float))); steps = ts["step"]
            if ps:
                ax.plot(steps / steps[-1], np.mean(ps, 0), "-o", ms=2.5, label=m, color=cols[m])
        ax.axhline(.9, ls=":", c="k", lw=.7); ax.axhline(.5, ls=":", c="k", lw=.7)
        ax.set_xscale("symlog", linthresh=0.02); ax.set_title(f"{ds} (D=16 probe/none)")
        ax.set_xlabel("fraction of training (symlog)"); ax.set_ylim(-0.2, 1.3)
    axs[0].set_ylabel("baseline-normalised progress  (x_t-x_0)/(x_T-x_0)"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "fig_timing_progress.png"), dpi=130)


if __name__ == "__main__":
    main()
