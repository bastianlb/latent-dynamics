"""Dependence of the class-wise generator spectra on the SGD noise scale nu = lr/B (block G + lrB).   Usage: python vfield_spectra_nu.py [--out DIR]
Fit  log|metric| = a + b log lr + c log nu  at matched t = lr*step bands (spectral quantities per step are divided by lr to give per-unit-t rates), N in {1000, 4000}, wd=0, lines: G (B=128), nu_fixed, lr_fixed."""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import argparse, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/spectra"


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X]); beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta; dof = max(len(y) - X.shape[1], 1); s2 = (res ** 2).sum() / dof
    cov = s2 * np.linalg.pinv(X.T @ X); r2 = 1 - (res ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-30); return beta, np.sqrt(np.diag(cov)), r2


def line_of(fam, lr, B):
    if fam == "sgdlr": return "G"
    nu = lr / B; return "nu_fixed" if abs(nu - 0.025 / 128) < 1e-9 else ("lr_fixed" if abs(lr - 0.025) < 1e-9 else "other")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args(); out = a.out
    rows = [r for r in csv.DictReader(open(f"{out}/summary_runs.csv")) if r["family"] in ("sgdlr", "lrB") and r["probe"] == "test" and float(r["wd"]) == 0.0 and int(r["N"]) in (1000, 4000)]
    f = lambda x: float(x) if x not in ("", None) else np.nan; res = []
    for N in (1000, 4000):
        for gen in ("res", "tot"):
            for m, per_t, logabs in (("abscissa_med", True, True), ("n_exp", False, False), ("n_cplx", False, False), ("logcond", False, False), ("jump", False, True), ("trace_over_d", True, True), ("gap12", True, True)):
                for band in ("t10_100", "t100_1000"):
                    rs = [r for r in rows if int(r["N"]) == N and np.isfinite(f(r[f"{m}_{gen}_{band}"]))]
                    if len(rs) < 8: continue
                    y = np.array([f(r[f"{m}_{gen}_{band}"]) / (f(r["lr"]) if per_t else 1.0) for r in rs]); lr = np.array([f(r["lr"]) for r in rs]); nu = np.array([f(r["nu"]) for r in rs])
                    if logabs: y = np.log(np.abs(y) + 1e-30)
                    elif m in ("n_exp", "n_cplx"): pass
                    if len({r["family"] if r["family"] == "sgdlr" else "lrB" for r in rs}) < 2: continue
                    X = np.column_stack([np.log(lr), np.log(nu)]); b, se, r2 = ols(X, y)
                    res.append(dict(N=N, generator=gen, metric=m + ("(log|.|/lr)" if (logabs and per_t) else ("(log|.|)" if logabs else "")), band=band, n=len(rs), b_logLR=b[1], se_b=se[1], c_logNU=b[2], se_c=se[2], R2=r2, verdict=("both" if (abs(b[1]) > 2 * se[1] and abs(b[2]) > 2 * se[2]) else "lr" if abs(b[1]) > 2 * se[1] else "nu" if abs(b[2]) > 2 * se[2] else "neither")))
    cols = list(res[0]) if res else []
    with open(f"{out}/nu_dependence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(res)
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), squeeze=False); col = {"G": "C2", "nu_fixed": "C0", "lr_fixed": "C1", "other": "gray"}
    for i, N in enumerate((1000, 4000)):
        for j, (m, lab) in enumerate((("abscissa_med_res_t100_1000", "abscissa of Lres per unit t (100<t<1000)"), ("n_exp_res_t100_1000", "n expanding (Lres)"), ("n_cplx_res_t100_1000", "n complex pairs"), ("logcond_res_t100_1000", "log cond eigvec"))):
            for line in ("G", "nu_fixed", "lr_fixed"):
                rs = [r for r in rows if int(r["N"]) == N and line_of(r["family"], f(r["lr"]), int(r["B"])) == line and np.isfinite(f(r[m]))]
                ax[i, j].scatter([f(r["nu"]) for r in rs], [f(r[m]) / (f(r["lr"]) if m.startswith("abscissa") else 1.0) for r in rs], c=col[line], s=14, label=line if (i == 0 and j == 0) else None)
            ax[i, j].set_xscale("log"); ax[i, j].set_xlabel("nu = lr/B"); ax[i, j].set_title(f"N={N}: {lab}", fontsize=8)
            if m.startswith("abscissa"): ax[i, j].set_yscale("symlog", linthresh=1e-5)
    ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{out}/figures/nu_dependence.png", dpi=60); plt.close(fig); print("done", len(res))


if __name__ == "__main__": main()
