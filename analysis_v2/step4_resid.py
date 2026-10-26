#!/usr/bin/env python
"""A3: ceiling-free targets. (a) within-config seed residual; (b) design-residual target. usage: --dataset mnist|fashion"""
import step4_lib as L
import argparse
import numpy as np
from scipy import stats
S = L.S
TARGETS = ["acc_15kw", "gap_15kw"]


def center(X, g):
    Xc = X.copy()
    for c in np.unique(g):
        m = g == c
        Xc[m] = X[m] - X[m].mean(0)
    return Xc


def design_basis(Z, kind, mu=None, sd=None):
    if mu is None:
        mu, sd = Z.mean(0), Z.std(0); sd[sd < 1e-12] = 1
    Zs = (Z - mu) / sd
    cols = [np.ones((len(Z), 1)), Zs]
    if kind == "quad":
        k = Zs.shape[1]
        cols.append(Zs ** 2)
        cols += [(Zs[:, i] * Zs[:, j])[:, None] for i in range(k) for j in range(i + 1, k)]
    return np.hstack(cols), mu, sd


def design_fit_predict(Ztr, ytr, Zte, kind):
    A, mu, sd = design_basis(Ztr, kind)
    lam = 1e-6 if kind == "lin" else 1e-2
    R = lam * np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ ytr)
    Ate, _, _ = design_basis(Zte, kind, mu, sd)
    return A @ w, Ate @ w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist")
    ap.parse_args()
    runs = L.get_runs()
    rows = []
    for tname in TARGETS:
        y = L.hasher(runs, tname)
        for cut in L.CUTS:
            B0, DES, G, g, seeds, names, EX = L.build_all(runs, cut, with_ext=False)
            # ---- (a) within-config seed residual
            ycs = center(y[:, None], g)[:, 0]
            nseed = np.array([(g == c).sum() for c in np.unique(g)])
            sd_within = float(np.sqrt(np.mean([np.var(y[g == c], ddof=1) for c in np.unique(g) if (g == c).sum() > 1])))
            Xb = center(B0, g); Xc = np.hstack([Xb, center(G, g)])
            pb = S.loco(Xb, ycs, g, "ridge"); pc = S.loco(Xc, ycs, g, "ridge")
            pg = S.loco(center(G, g), ycs, g, "ridge")
            s = L.summarize(ycs, pb, pc, g, tname)
            rows.append(dict(analysis="3a_within_config", variant="B0c -> B0c+Gc", target=tname, cutoff=cut, n_runs=len(y), n_configs=len(np.unique(g)),
                             min_seeds=int(nseed.min()), within_cfg_sd=sd_within, r2_base=S.r2(ycs, pb), r2_comp=S.r2(ycs, pc), r2_G=S.r2(ycs, pg),
                             spearman_base=stats.spearmanr(ycs, pb)[0], spearman_comp=stats.spearmanr(ycs, pc)[0], **s))
            # ---- (b) design-residual target (training-fold design model)
            for kind in ("lin", "quad"):
                rt = np.zeros(len(y)); ptb = np.zeros(len(y)); ptc = np.zeros(len(y)); ptg = np.zeros(len(y))
                XB, XC = B0, np.hstack([B0, G])
                for c in np.unique(g):
                    te = g == c; tr = ~te
                    rtr, rte_fit = design_fit_predict(DES[tr], y[tr], DES[te], kind)
                    rtr_res = y[tr] - rtr
                    rt[te] = y[te] - rte_fit
                    ptb[te] = S.ridge_fit_predict(XB[tr], rtr_res, g[tr], XB[te])
                    ptc[te] = S.ridge_fit_predict(XC[tr], rtr_res, g[tr], XC[te])
                    ptg[te] = S.ridge_fit_predict(G[tr], rtr_res, g[tr], G[te])
                s = L.summarize(rt, ptb, ptc, g, tname)
                rows.append(dict(analysis="3b_design_residual", variant=f"{kind}: B0 -> B0+G", target=tname, cutoff=cut, n_runs=len(y), n_configs=len(np.unique(g)),
                                 resid_sd=float(rt.std()), r2_base=S.r2(rt, ptb), r2_comp=S.r2(rt, ptc), r2_G=S.r2(rt, ptg),
                                 spearman_base=stats.spearmanr(rt, ptb)[0], spearman_comp=stats.spearmanr(rt, ptc)[0], **s))
            print(tname, cut, "done", flush=True)
    L.write_csv(f"{L.OUT2}/step4_resid_results.csv", rows)
    lines = []
    for an in ("3a_within_config", "3b_design_residual"):
        for var in dict.fromkeys(r["variant"] for r in rows if r["analysis"] == an):
            for t in TARGETS:
                sub = sorted([r for r in rows if r["analysis"] == an and r["variant"] == var and r["target"] == t], key=lambda r: r["cutoff"])
                lines.append(f"[{an}] {var} {t}  " + (f"within-config SD {sub[0]['within_cfg_sd']:.4f}, min seeds {sub[0]['min_seeds']}" if an.startswith("3a") else f"resid SD {sub[0]['resid_sd']:.4f}"))
                lines.append("   cut      " + " ".join(f"{r['cutoff']:>14d}" for r in sub))
                lines.append("   R2 base  " + " ".join(f"{r['r2_base']:+14.3f}" for r in sub))
                lines.append("   R2 +G    " + " ".join(f"{r['r2_comp']:+14.3f}" for r in sub))
                lines.append("   R2 G only" + " ".join(f"{r['r2_G']:+14.3f}" for r in sub))
                lines.append("   dR2      " + " ".join(f"{r['delta']:+14.3f}" for r in sub))
                lines.append("   95%CI    " + " ".join(f"[{r['lo95']:+.3f},{r['hi95']:+.3f}]" for r in sub))
                lines.append("   class    " + " | ".join(r["label"] for r in sub))
    open(f"{L.OUT2}/step4_resid_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
