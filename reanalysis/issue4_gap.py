#!/usr/bin/env python
"""Generalisation-gap vs geometry/topology with correct units (8 configs x 3 seeds, AE excluded).

Gap definitions: 'snap' = snapshot train_acc - test_acc at final step (paper; train_acc is the running
train-mode mean since the previous snapshot), 'eval' = eval-mode full-train accuracy of model_final.pt -
test accuracy (final_train_eval.py).
Outputs: gap_stats.csv, gap_ranges.csv, fig_gap_config_level.png, issue4_summary.txt
"""
from __future__ import annotations
import common as C
from stats_util import boot_ci_corr, power_r, partial_corr

import csv, os
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARS = {"fisher": "fisher_ratio", "H1_L1": "h1_lifetime_sum_alpha1.0", "H1_ent": "h1_entropy", "pmag": "pmag",
        "rawnorm": "raw_mean_norm", "logD": None, "test_acc": "test_acc"}
lines = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)


def main():
    ev = {r["run_id"]: r for r in csv.DictReader(open(os.path.join(C.OUT, "final_train_eval.csv")))}
    allruns = [r for r in C.list_runs() if r["objective"] != "autoencoder"]
    data = {}
    for r in allruns:
        ts = C.load_timeseries(r["run_id"]); e = ev[r["run_id"]]
        d = dict(r); d["gap_snap"] = ts["train_acc"][-1] - ts["test_acc"][-1]; d["gap_eval"] = float(e["gap_eval"])
        for k, c in VARS.items():
            d[k] = np.log2(r["D"]) if k == "logD" else ts[c][-1]
        data.setdefault(r["dataset"], []).append(d)

    stat_rows, range_rows = [], []
    P("n_configs=8, seeds=3, power: min |r| detectable at 80%% power (alpha .05, two-sided): n=8 -> %.2f ; n=24 (if runs were independent) -> %.2f ; n=72 -> %.2f" %
      (power_r(8), power_r(24), power_r(72)))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for gi, gname in enumerate(("gap_snap", "gap_eval")):
        for ci, ds in enumerate(C.DATASETS):
            rows = data[ds]; cfgs = sorted({(r["objective"], r["norm"], r["D"]) for r in rows})
            cid = np.array([cfgs.index((r["objective"], r["norm"], r["D"])) for r in rows])
            g = np.array([r[gname] for r in rows])
            gm = np.array([g[cid == i].mean() for i in range(len(cfgs))])
            within_sd = np.sqrt(np.mean([g[cid == i].var(ddof=1) for i in range(len(cfgs))]))
            P(f"\n=== {ds} [{gname}] runs n={len(rows)}, configs n={len(cfgs)}: gap range [{g.min():+.4f},{g.max():+.4f}], SD(runs)={g.std(ddof=1):.4f}, "
              f"SD(config means)={gm.std(ddof=1):.4f}, within-config seed SD={within_sd:.4f}")
            range_rows.append(dict(dataset=ds, gap=gname, min=g.min(), max=g.max(), sd_runs=g.std(ddof=1), sd_config_means=gm.std(ddof=1), seed_sd=within_sd))
            for k in VARS:
                x = np.array([r[k] for r in rows]); xm = np.array([x[cid == i].mean() for i in range(len(cfgs))])
                r_run, p_run = stats.pearsonr(x, g)                      # paper's (pseudo-replicated) number
                r_cfg, p_cfg = stats.pearsonr(xm, gm)
                s_cfg, ps_cfg = stats.spearmanr(xm, gm)
                lo, hi = boot_ci_corr(xm, gm, n=4000)                     # resample configs
                clo, chi = boot_ci_corr(x, g, n=4000, clusters=cid)       # cluster bootstrap (configs as clusters) on run-level r
                # permutation p-value for config-level pearson
                rng = np.random.default_rng(1)
                perm = np.array([stats.pearsonr(xm, rng.permutation(gm))[0] for _ in range(5000)])
                p_perm = (np.sum(np.abs(perm) >= abs(r_cfg)) + 1) / (len(perm) + 1)
                stat_rows.append(dict(dataset=ds, gap=gname, var=k, r_run=r_run, p_run_naive=p_run, r_cfg=r_cfg, p_cfg=p_cfg, p_perm=p_perm,
                                      boot_lo=lo, boot_hi=hi, spearman_cfg=s_cfg, p_spearman_cfg=ps_cfg, clusterboot_lo=clo, clusterboot_hi=chi))
                P(f"  {k:8s} run-level r={r_run:+.2f} (naive p={p_run:.2f}) | config-level r={r_cfg:+.2f} p={p_cfg:.2f} perm-p={p_perm:.2f} "
                  f"boot95=[{lo:+.2f},{hi:+.2f}] spearman={s_cfg:+.2f} (p={ps_cfg:.2f}) | cluster-boot(run r) 95%=[{clo:+.2f},{chi:+.2f}]")
            if gname == "gap_snap" or gname == "gap_eval":
                ax = axs[gi, ci]
                x = np.array([r["H1_L1"] for r in rows]); xm = np.array([x[cid == i].mean() for i in range(len(cfgs))])
                ax.scatter(x, g, s=10, c="0.6"); ax.scatter(xm, gm, s=50, c="C3")
                for i, c in enumerate(cfgs): ax.annotate(f"{c[0][:6]}/{c[1]}/{c[2]}", (xm[i], gm[i]), fontsize=6)
                ax.set_title(f"{ds} [{gname}]: runs (grey), config means (red)"); ax.set_xlabel("final H1 lifetime-sum"); ax.set_ylabel("gap")
    fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "fig_gap_config_level.png"), dpi=110)

    # config table + partial correlations controlling for log2 D and for test_acc (run level, n=24)
    for gname in ("gap_snap", "gap_eval"):
        for ds in C.DATASETS:
            rows = data[ds]; g = np.array([r[gname] for r in rows])
            P(f"\n[{ds} {gname}] partial Spearman (run-level, n=24) controlling log2 D: " + "  ".join(
                f"{k}:{partial_corr([r[k] for r in rows], g, [[r['logD']] for r in rows]):+.2f}" for k in ("fisher", "H1_L1", "H1_ent", "pmag")) +
                "   controlling log2 D + test_acc: " + "  ".join(f"{k}:{partial_corr([r[k] for r in rows], g, [[r['logD'], r['test_acc']] for r in rows]):+.2f}" for k in ("fisher", "H1_L1", "H1_ent", "pmag")))
    for ds in C.DATASETS:
        rows = data[ds]
        P(f"\n[{ds}] config means: config | gap_snap | gap_eval | test_acc | fisher | H1_L1")
        for c in sorted({(r["objective"], r["norm"], r["D"]) for r in rows}):
            rr = [r for r in rows if (r["objective"], r["norm"], r["D"]) == c]
            P(f"   {c[0]}/{c[1]}/d{c[2]:<3d} {np.mean([r['gap_snap'] for r in rr]):+.4f} {np.mean([r['gap_eval'] for r in rr]):+.4f} {np.mean([r['test_acc'] for r in rr]):.3f} {np.mean([r['fisher'] for r in rr]):6.2f} {np.mean([r['H1_L1'] for r in rr]):6.1f}")
    # pooled config-level (dataset-standardised), n = 24 config means
    for gname in ("gap_snap", "gap_eval"):
        for k in ("fisher", "H1_L1", "pmag"):
            X, Y = [], []
            for ds in C.DATASETS:
                rows = data[ds]; cfgs = sorted({(r["objective"], r["norm"], r["D"]) for r in rows})
                for c in cfgs:
                    rr = [r for r in rows if (r["objective"], r["norm"], r["D"]) == c]
                    X.append((ds, np.mean([r[k] for r in rr]), np.mean([r[gname] for r in rr])))
            xs = np.array([[x[1], x[2]] for x in X]); dsn = np.array([x[0] for x in X])
            for ds in C.DATASETS:
                m = dsn == ds; xs[m] = (xs[m] - xs[m].mean(0)) / xs[m].std(0)
            r, p = stats.pearsonr(xs[:, 0], xs[:, 1]); lo, hi = boot_ci_corr(xs[:, 0], xs[:, 1], n=4000)
            P(f"pooled config means (n=24, within-dataset z-scored) {gname} vs {k}: r={r:+.2f} p={p:.2f} boot95=[{lo:+.2f},{hi:+.2f}]")
    with open(os.path.join(C.OUT, "gap_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0])); w.writeheader(); w.writerows(stat_rows)
    with open(os.path.join(C.OUT, "gap_ranges.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(range_rows[0])); w.writeheader(); w.writerows(range_rows)
    open(os.path.join(C.OUT, "issue4_summary.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
