#!/usr/bin/env python
"""A8: PMag original-window vs dataset-shared-window (column B) Spearman across datasets, from reanalysis_v2/pmag_windows_final.csv."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "2"
import csv, numpy as np
from scipy import stats
SRC = "/vol/bitbucket/lbastian/latent-dynamics/reanalysis_v2/pmag_windows_final.csv"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/ext2"; os.makedirs(OUT + "/mnist", exist_ok=True)
rows = list(csv.DictReader(open(SRC)))
rng = np.random.default_rng(0)
res, lines = [], []
for ds in ("mnist", "fashion_mnist", "cifar10"):
    sel = [r for r in rows if r["dataset"] == ds]
    cfg = np.array([f"{r['objective']}/{r['norm']}/D{r['D']}" for r in sel])
    x = np.array([float(r["pmag_csv"]) for r in sel])
    for k, lab in (("B", "dataset-shared window (B)"), ("A", "run's own final window (A)"), ("Wd", "Wd"), ("U", "U"), ("mag_t1", "Mag(t=1)")):
        y = np.array([float(r[k]) for r in sel])
        rho = stats.spearmanr(x, y)[0]
        u = np.unique(cfg); bs = []
        for _ in range(4000):
            ids = np.concatenate([np.where(cfg == c)[0] for c in rng.choice(u, len(u))])
            if np.ptp(x[ids]) > 0 and np.ptp(y[ids]) > 0: bs.append(stats.spearmanr(x[ids], y[ids])[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res.append(dict(dataset=ds, variant=k, n_runs=len(sel), n_configs=len(u), rho=rho, ci_lo=lo, ci_hi=hi))
        lines.append(f"{ds:14s} orig vs {lab:28s} rho={rho:+.3f} [{lo:+.2f},{hi:+.2f}] n_runs={len(sel)} configs={len(u)}")
    # rho restricted to the 5 probe/none configs? (D=16 probe/none: 15 runs across seeds) -> also report
    m = np.array([(r["objective"], r["norm"]) == ("probe", "none") for r in sel])
    lines.append(f"{'':14s}   (probe/none, D in 2..32 only, n={m.sum()} runs: rho orig vs B = {stats.spearmanr(x[m], np.array([float(r['B']) for r in sel])[m])[0]:+.3f})")
with open(OUT + "/mnist/step4_pmag_windows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(res[0])); w.writeheader(); w.writerows(res)
open(OUT + "/mnist/step4_pmag_windows.txt", "w").write("\n".join(lines)); print("\n".join(lines))
