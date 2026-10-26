#!/usr/bin/env python
"""PMag scale-window artifact test.

Uses Mag(t) on a fixed log grid (compute_extra cache) of the mean-pairwise-normalised latents and
re-integrates PMag = mean_{[tmin,tmax]} Mag(t) (12 linearly spaced t, trapezoid, as in metrics.pmag)
under different windows:
  orig  per-snapshot window [1/d95, 1/d5]   (validated against timeseries.csv pmag)
  A     per-run FIXED window from that run's FINAL snapshot, applied to all its snapshots
  B     per-dataset GLOBAL window = geometric mean of the runs' final windows, shared by all runs
  W     wide per-dataset window  [min final tmin, max final tmax] over runs
  U     universal window [0.5, 10] in mean-distance units, all datasets
plus Mag(t) at fixed t in {1,2,4,8}.
Outputs: pmag_windows_final.csv, pmag_l2_ratio.csv, pmag_trajectory_shape.csv, fig_pmag_*.png
"""
from __future__ import annotations
import common as C

import csv, os
import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lines = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)


def mag_interp(t_grid, mag):
    """Cubic spline of log Mag vs log t -> callable on t."""
    cs = CubicSpline(np.log(t_grid), np.log(np.maximum(mag, 1e-12)))
    return lambda t: np.exp(cs(np.log(np.clip(t, t_grid[0], t_grid[-1]))))


def pmag_window(t_grid, mags, tmin, tmax, n=12):
    """mags [n_snap, n_t]; tmin,tmax scalar or [n_snap]."""
    tmin = np.broadcast_to(tmin, (len(mags),)); tmax = np.broadcast_to(tmax, (len(mags),))
    out = np.empty(len(mags))
    for i in range(len(mags)):
        ts = np.linspace(tmin[i], tmax[i], n)
        out[i] = np.trapezoid(mag_interp(t_grid, mags[i])(ts), ts) / (ts[-1] - ts[0])
    return out


def mag_at(t_grid, mags, t):
    return np.array([mag_interp(t_grid, m)(np.array([t]))[0] for m in mags])


def main():
    runs = C.list_runs(); caches = {r["run_id"]: C.load_cache(r["run_id"]) for r in runs}
    assert all(c is not None for c in caches.values()), "run compute_extra.py first"
    lv = list(caches[runs[0]["run_id"]]["d_q_levels"]); i5, i95 = lv.index(5), lv.index(95)
    W = {}  # (run_id) -> dict of series
    glob_win = {}
    for ds in C.DATASETS:
        fin = {r["run_id"]: (1 / caches[r["run_id"]]["d_q"][-1, i95], 1 / caches[r["run_id"]]["d_q"][-1, i5]) for r in runs if r["dataset"] == ds}
        tmins = np.array([v[0] for v in fin.values()]); tmaxs = np.array([v[1] for v in fin.values()])
        glob_win[ds] = dict(B=(np.exp(np.log(tmins).mean()), np.exp(np.log(tmaxs).mean())), W=(tmins.min(), tmaxs.max()), U=(0.5, 10.0))
        P(f"[{ds}] per-run FINAL windows: tmin in [{tmins.min():.2f},{tmins.max():.2f}] (median {np.median(tmins):.2f}), tmax in [{tmaxs.min():.2f},{tmaxs.max():.2f}] (median {np.median(tmaxs):.2f}); "
          f"B=({glob_win[ds]['B'][0]:.2f},{glob_win[ds]['B'][1]:.2f})")
    maxerr = 0
    for r in runs:
        c = caches[r["run_id"]]; tg, mg, dq = c["mag_t"], c["mag"], c["d_q"]; ds = r["dataset"]
        ts = C.load_timeseries(r["run_id"])
        orig = pmag_window(tg, mg, 1 / dq[:, i95], 1 / dq[:, i5])
        maxerr = max(maxerr, np.max(np.abs(orig - ts["pmag"]) / ts["pmag"]))
        series = dict(pmag_csv=ts["pmag"], pmag_orig_cache=orig,
                      A=pmag_window(tg, mg, 1 / dq[-1, i95], 1 / dq[-1, i5]),
                      B=pmag_window(tg, mg, *glob_win[ds]["B"]), Wd=pmag_window(tg, mg, *glob_win[ds]["W"]),
                      U=pmag_window(tg, mg, *glob_win["mnist"]["U"]))
        for t in (1, 2, 4, 8): series[f"mag_t{t}"] = mag_at(tg, mg, t)
        W[r["run_id"]] = series
    P(f"validation: max relative error of cache-reintegrated per-snapshot-window PMag vs timeseries.csv pmag = {maxerr:.3%}")

    # ---------- final-step table + l2/none ratios ----------
    keys = ["pmag_csv", "A", "B", "Wd", "U", "mag_t1", "mag_t2", "mag_t4", "mag_t8"]
    rows = []
    for r in runs:
        d = dict(run_id=r["run_id"], dataset=r["dataset"], objective=r["objective"], norm=r["norm"], D=r["D"], seed=r["seed"])
        d.update({k: W[r["run_id"]][k][-1] for k in keys}); rows.append(d)
    with open(os.path.join(C.OUT, "pmag_windows_final.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    P("\nFinal-step PMag variants, D=16 configs, mean over 3 seeds; ratio = config / probe-none")
    ratio_rows = []
    for ds in C.DATASETS:
        base = {k: np.mean([d[k] for d in rows if d["dataset"] == ds and (d["objective"], d["norm"], d["D"]) == ("probe", "none", 16)]) for k in keys}
        for cfg in (("probe", "none"), ("probe", "l2"), ("probe", "bn"), ("probe_recon", "none"), ("autoencoder", "none")):
            v = {k: np.mean([d[k] for d in rows if d["dataset"] == ds and (d["objective"], d["norm"], d["D"]) == (*cfg, 16)]) for k in keys}
            ratio_rows.append(dict(dataset=ds, config="/".join(cfg), **{k: v[k] for k in keys}, **{k + "_ratio": v[k] / base[k] for k in keys}))
            P(f"  {ds:13s} {'/'.join(cfg):18s} " + "  ".join(f"{k}={v[k]:7.2f}(x{v[k]/base[k]:.2f})" for k in keys))
    with open(os.path.join(C.OUT, "pmag_l2_ratio.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ratio_rows[0])); w.writeheader(); w.writerows(ratio_rows)

    # ---------- across-all-runs rank agreement of variants with original ----------
    P("\nSpearman rho of final-step PMag variant vs original PMag across 27 runs per dataset:")
    for ds in C.DATASETS:
        sel = [d for d in rows if d["dataset"] == ds]
        P(f"  {ds}: " + "  ".join(f"{k}:{stats.spearmanr([d['pmag_csv'] for d in sel],[d[k] for d in sel])[0]:+.2f}" for k in keys[1:]))
        P(f"      rho vs Fisher: orig={stats.spearmanr([d['pmag_csv'] for d in sel],[C.load_timeseries(d['run_id'])['fisher_ratio'][-1] for d in sel])[0]:+.2f}  " +
          "  ".join(f"{k}:{stats.spearmanr([C.load_timeseries(d['run_id'])['fisher_ratio'][-1] for d in sel],[d[k] for d in sel])[0]:+.2f}" for k in keys[1:]))

    # ---------- trajectory shape ----------
    P("\nTrajectory shape, D=16 probe/none and probe/l2 (mean over seeds): per-run Spearman(orig PMag(t), variant(t)) and p_first / t90 under each variant")
    shape_rows = []
    for ds in C.DATASETS:
        for cfg in (("probe", "none"), ("probe", "l2"), ("probe", "bn"), ("probe_recon", "none"), ("autoencoder", "none")):
            rr = [r for r in runs if r["dataset"] == ds and (r["objective"], r["norm"], r["D"]) == (*cfg, 16)]
            for k in ["pmag_csv", "A", "B", "Wd", "U", "mag_t1", "mag_t2", "mag_t4", "mag_t8"]:
                rho = np.mean([stats.spearmanr(W[r["run_id"]]["pmag_csv"], W[r["run_id"]][k])[0] for r in rr])
                ps = []; t90 = []; mono = []
                for r in rr:
                    x = W[r["run_id"]][k]; steps = C.load_timeseries(r["run_id"])["step"]; p = (x - x[0]) / (x[-1] - x[0]) if abs(x[-1] - x[0]) > 1e-9 else np.full_like(x, np.nan)
                    ps.append(p[1]); idx = np.where(p >= .9)[0]; t90.append(steps[idx[0]] / steps[-1] if len(idx) else np.nan)
                    mono.append(x[-1] / x[0])
                shape_rows.append(dict(dataset=ds, config="/".join(cfg), variant=k, rho_vs_orig=rho, p_first=np.mean(ps), t90=np.nanmean(t90), final_over_initial=np.mean(mono)))
    with open(os.path.join(C.OUT, "pmag_trajectory_shape.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(shape_rows[0])); w.writeheader(); w.writerows(shape_rows)
    for s in shape_rows:
        if s["config"] in ("probe/none", "probe/l2"):
            P(f"  {s['dataset']:13s} {s['config']:11s} {s['variant']:9s} rho_vs_orig={s['rho_vs_orig']:+.2f}  final/initial={s['final_over_initial']:.2f}  p_first={s['p_first']:+.2f}  t90={s['t90']:.2f}")

    # ---------- figure ----------
    fig, axs = plt.subplots(3, 4, figsize=(18, 10))
    for ci, ds in enumerate(C.DATASETS):
        for vi, (k, title) in enumerate((("pmag_csv", "orig (per-snapshot window)"), ("A", "A: run's final window"), ("B", "B: dataset-global window"), ("mag_t2", "Mag(t=2)"))):
            ax = axs[ci, vi]
            for cfg, col in ((("probe", "none"), "C0"), (("probe", "l2"), "C2"), (("probe", "bn"), "C4"), (("probe_recon", "none"), "C1"), (("autoencoder", "none"), "C3")):
                rr = [r for r in runs if r["dataset"] == ds and (r["objective"], r["norm"], r["D"]) == (*cfg, 16)]
                steps = C.load_timeseries(rr[0]["run_id"])["step"]
                ax.plot(steps, np.mean([W[r["run_id"]][k] for r in rr], 0), c=col, label="/".join(cfg))
            ax.set_title(f"{ds}: {title}"); ax.set_xscale("symlog", linthresh=50); ax.set_yscale("log")
            if ci == 0 and vi == 0: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "fig_pmag_windows_trajectories.png"), dpi=100)
    open(os.path.join(C.OUT, "issue3_summary.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
