#!/usr/bin/env python
"""Is topology (H1) redundant with geometry (Fisher)?  Sections 2a-2e of the reanalysis.

2a final-step Spearman matrices per dataset (27 runs, and excl. AE)
2b within-run first-differenced correlations over time
2c partial correlations of H1 / PMag with Fisher controlling for log raw norm and D
2d concrete disagreement cases
2e H1 noise: lifetime mass below percentile/threshold, top-k bars (needs compute_extra cache)
"""
from __future__ import annotations
import common as C
from stats_util import partial_corr

import csv, os, itertools
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = ["test_acc", "fisher_ratio", "h1_lifetime_sum_alpha1.0", "h1_entropy", "pmag", "raw_mean_norm",
        "h0_lifetime_sum_alpha1.0", "h0_entropy", "separation_mean", "compactness_centroid"]
SHORT = {"test_acc": "acc", "fisher_ratio": "fisher", "h1_lifetime_sum_alpha1.0": "H1_L1", "h1_entropy": "H1_ent",
         "pmag": "pmag", "raw_mean_norm": "rawnorm", "h0_lifetime_sum_alpha1.0": "H0_L1", "h0_entropy": "H0_ent",
         "separation_mean": "sep", "compactness_centroid": "compact", "offline_probe_acc": "offacc",
         "h1_top5": "H1_top5", "h1_top10": "H1_top10", "h1_top20": "H1_top20", "D": "D"}
out_lines = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); out_lines.append(s)


def spearman_matrix(X, names):
    k = len(names); R = np.eye(k)
    for i, j in itertools.combinations(range(k), 2):
        R[i, j] = R[j, i] = stats.spearmanr(X[:, i], X[:, j])[0]
    return R


def final_table(runs):
    rows = []
    for r in runs:
        ts = C.load_timeseries(r["run_id"])
        d = dict(r); d.update({c: ts[c][-1] for c in COLS})
        ca = C.load_cache(r["run_id"])
        if ca is not None:
            d["offline_probe_acc"] = ca["offline_probe_acc"][-1]
        rows.append(d)
    return rows


def main():
    os.makedirs(C.OUT, exist_ok=True)
    allruns = C.list_runs()
    # ---------------- 2a ----------------
    P("## 2a final-step Spearman correlations")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    final_all = {}
    for ax, ds in zip(axs, C.DATASETS):
        rows = final_table([r for r in allruns if r["dataset"] == ds]); final_all[ds] = rows
        names = [c for c in COLS] + (["offline_probe_acc"] if all("offline_probe_acc" in r for r in rows) else [])
        X = np.array([[r[c] for c in names] for r in rows]); lab = [SHORT[c] for c in names]
        R = spearman_matrix(X, names)
        with open(os.path.join(C.OUT, f"corr_final_{ds}.csv"), "w") as f:
            f.write("," + ",".join(lab) + "\n")
            for i, l in enumerate(lab): f.write(l + "," + ",".join(f"{v:.3f}" for v in R[i]) + "\n")
        im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r"); ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=60)
        ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab); ax.set_title(f"{ds}: Spearman, final step, n={len(rows)}")
        for i in range(len(lab)):
            for j in range(len(lab)): ax.text(j, i, f"{R[i,j]:.1f}", ha="center", va="center", fontsize=6)
        iF, iH, iP = names.index("fisher_ratio"), names.index("h1_lifetime_sum_alpha1.0"), names.index("pmag")
        iE = names.index("h1_entropy")
        isae = np.array([r["objective"] == "autoencoder" for r in rows])
        P(f"[{ds}] rho(Fisher,H1_L1)  all27={R[iF,iH]:+.3f}  excl AE(n={(~isae).sum()})={stats.spearmanr(X[~isae,iF],X[~isae,iH])[0]:+.3f};"
          f"  rho(Fisher,H1_ent)={R[iF,iE]:+.3f}; rho(Fisher,PMag)={R[iF,iP]:+.3f} (exclAE {stats.spearmanr(X[~isae,iF],X[~isae,iP])[0]:+.3f});"
          f" rho(H1_L1,PMag)={R[iH,iP]:+.3f}")
        # within-objective restrictions: probe/none capacity sweep only (15 runs) and D=16 only
        cap = np.array([r["objective"] == "probe" and r["norm"] == "none" for r in rows])
        P(f"      capacity sweep only (n={cap.sum()}): rho(Fisher,H1_L1)={stats.spearmanr(X[cap,iF],X[cap,iH])[0]:+.3f}, "
          f"rho(Fisher,PMag)={stats.spearmanr(X[cap,iF],X[cap,iP])[0]:+.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "fig_corr_matrices_final.png"), dpi=110)

    # ---------------- 2b ----------------
    P("\n## 2b within-run correlations over time (first-differenced; Spearman over snapshots), per run then summarised")
    rows_b = []
    for r in allruns:
        ts = C.load_timeseries(r["run_id"]); steps = ts["step"]
        f, h, p, n = ts["fisher_ratio"], ts["h1_lifetime_sum_alpha1.0"], ts["pmag"], ts["raw_mean_norm"]
        for tag, sel in (("all", slice(1, None)), ("late(step>=10%T)", steps[1:] >= 0.1 * steps[-1])):
            def dcorr(a, b, logd):
                da = np.diff(np.log(a) if logd else a)[sel]; db = np.diff(np.log(b) if logd else b)[sel]
                return stats.spearmanr(da, db)[0]
            rows_b.append(dict(run_id=r["run_id"], dataset=r["dataset"], objective=r["objective"], norm=r["norm"], D=r["D"], window=tag,
                               lvl_F_H1=stats.spearmanr(f, h)[0], d_F_H1=dcorr(f, h, False), dlog_F_H1=dcorr(f, h, True),
                               d_F_pmag=dcorr(f, p, False), d_H1_pmag=dcorr(h, p, False), dlog_F_pmag=dcorr(f, p, True)))
    with open(os.path.join(C.OUT, "within_run_corr.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_b[0])); w.writeheader(); w.writerows(rows_b)
    for ds in C.DATASETS:
        for win in ("all", "late(step>=10%T)"):
            rr = [r for r in rows_b if r["dataset"] == ds and r["window"] == win and r["objective"] != "autoencoder"]
            g = lambda k: np.array([r[k] for r in rr])
            P(f"[{ds} {win} n={len(rr)} non-AE] median rho: level(F,H1)={np.nanmedian(g('lvl_F_H1')):+.2f}  diff(F,H1)={np.nanmedian(g('d_F_H1')):+.2f} "
              f"[IQR {np.nanpercentile(g('d_F_H1'),25):+.2f},{np.nanpercentile(g('d_F_H1'),75):+.2f}]  dlog(F,H1)={np.nanmedian(g('dlog_F_H1')):+.2f}  "
              f"diff(F,pmag)={np.nanmedian(g('d_F_pmag')):+.2f}  diff(H1,pmag)={np.nanmedian(g('d_H1_pmag')):+.2f}  frac runs with diff rho(F,H1)>0: {(g('d_F_H1')>0).mean():.2f}")
    # histogram
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, win in zip(axs, ("all", "late(step>=10%T)")):
        for ds in C.DATASETS:
            v = [r["d_F_H1"] for r in rows_b if r["dataset"] == ds and r["window"] == win and r["objective"] != "autoencoder"]
            ax.hist(v, bins=np.linspace(-1, 1, 21), alpha=.5, label=ds)
        ax.set_title(f"within-run rho(dFisher,dH1_L1) [{win}], non-AE runs"); ax.legend(); ax.axvline(0, c="k", lw=.6)
    fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "fig_within_run_diff_corr.png"), dpi=110)

    # ---------------- 2c ----------------
    P("\n## 2c partial Spearman correlations at final step (controls: log raw_mean_norm, D)")
    for ds in C.DATASETS:
        rows = final_all[ds]
        for tag, sel in (("all27", lambda r: True), ("exclAE", lambda r: r["objective"] != "autoencoder")):
            rr = [r for r in rows if sel(r)]
            F = np.array([r["fisher_ratio"] for r in rr]); H = np.array([r["h1_lifetime_sum_alpha1.0"] for r in rr])
            PM = np.array([r["pmag"] for r in rr]); HE = np.array([r["h1_entropy"] for r in rr])
            Z = np.column_stack([np.log([r["raw_mean_norm"] for r in rr]), [r["D"] for r in rr]])
            P(f"[{ds} {tag} n={len(rr)}] raw rho(F,H1)={stats.spearmanr(F,H)[0]:+.2f} -> partial={partial_corr(F,H,Z):+.2f}; "
              f"raw rho(F,H1ent)={stats.spearmanr(F,HE)[0]:+.2f} -> partial={partial_corr(F,HE,Z):+.2f}; "
              f"raw rho(F,PMag)={stats.spearmanr(F,PM)[0]:+.2f} -> partial={partial_corr(F,PM,Z):+.2f}; "
              f"rho(H1,PMag)={stats.spearmanr(H,PM)[0]:+.2f} -> partial={partial_corr(H,PM,Z):+.2f}")
    # pooled over datasets with dataset-demeaned ranks
    rr = [r for ds in C.DATASETS for r in final_all[ds] if r["objective"] != "autoencoder"]
    ds_dummies = np.array([[r["dataset"] == d for d in C.DATASETS[1:]] for r in rr], float)
    Z = np.column_stack([np.log([r["raw_mean_norm"] for r in rr]), [r["D"] for r in rr], ds_dummies])
    F = np.array([r["fisher_ratio"] for r in rr]); H = np.array([r["h1_lifetime_sum_alpha1.0"] for r in rr]); PM = np.array([r["pmag"] for r in rr])
    P(f"[pooled non-AE n={len(rr)}, +dataset dummies] partial rho(F,H1)={partial_corr(F,H,Z):+.2f}  partial rho(F,PMag)={partial_corr(F,PM,Z):+.2f}")

    # ---------------- 2d ----------------
    P("\n## 2d disagreement cases")
    for ds in C.DATASETS:
        rows = final_all[ds]
        cand = []
        for a, b in itertools.combinations(rows, 2):
            if a["fisher_ratio"] <= 0 or b["fisher_ratio"] <= 0: continue
            rf = abs(np.log(a["fisher_ratio"] / b["fisher_ratio"]))
            rh = abs(np.log(a["h1_lifetime_sum_alpha1.0"] / b["h1_lifetime_sum_alpha1.0"]))
            if rf < 0.10 and a["seed"] != b["seed"] or rf < 0.10 and (a["objective"], a["norm"], a["D"]) != (b["objective"], b["norm"], b["D"]):
                cand.append((rh, rf, a, b))
        cand = [c for c in cand if (c[2]["objective"], c[2]["norm"], c[2]["D"]) != (c[3]["objective"], c[3]["norm"], c[3]["D"])]
        cand.sort(key=lambda c: -c[0])
        P(f"[{ds}] pairs of DIFFERENT configs with |ln Fisher ratio|<0.10, largest H1_L1 log-ratio:")
        for rh, rf, a, b in cand[:4]:
            P(f"   {a['run_id']}: F={a['fisher_ratio']:.2f} H1={a['h1_lifetime_sum_alpha1.0']:.1f} | {b['run_id']}: F={b['fisher_ratio']:.2f} H1={b['h1_lifetime_sum_alpha1.0']:.1f}  (H1 ratio x{np.exp(rh):.2f})")
        # capacity-sweep means: D vs F, H1
        P(f"   capacity means (probe/none) D: Fisher | H1_L1 | PMag")
        for D in (2, 4, 8, 16, 32):
            rr = [r for r in rows if r["objective"] == "probe" and r["norm"] == "none" and r["D"] == D]
            P(f"     D={D:2d}: {np.mean([r['fisher_ratio'] for r in rr]):.2f} | {np.mean([r['h1_lifetime_sum_alpha1.0'] for r in rr]):.1f} | {np.mean([r['pmag'] for r in rr]):.2f}")
    # intervals where F and H1 move in same direction (both up) with large changes
    P("   time windows: share of snapshot-intervals (late, step>=10%T, non-AE runs) with dFisher and dH1_L1 of the SAME sign (naive anti-correlation predicts opposite):")
    for ds in C.DATASETS:
        same = tot = 0
        ex = None
        for r in allruns:
            if r["dataset"] != ds or r["objective"] == "autoencoder": continue
            ts = C.load_timeseries(r["run_id"]); st = ts["step"][1:]; sel = st >= 0.1 * ts["step"][-1]
            df, dh = np.diff(ts["fisher_ratio"])[sel], np.diff(ts["h1_lifetime_sum_alpha1.0"])[sel]
            same += int((np.sign(df) == np.sign(dh)).sum()); tot += len(df)
        P(f"     {ds}: {same}/{tot} = {same/tot:.2f}")
    open(os.path.join(C.OUT, "issue2_ad_summary.txt"), "w").write("\n".join(out_lines))


if __name__ == "__main__":
    main()
