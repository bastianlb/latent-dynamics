#!/usr/bin/env python
"""Amendment 4 A4.5: design-residual targets with flexible design models (lin, quad, gbm) + saturated config-mean (= within-config residual). usage: --dataset"""
import step5_lib as M
import step4_resid as R
import argparse
from multiprocessing import Pool
import numpy as np
L, S = M.L, M.S
RUNS = None


def gbm_design(Ztr, ytr, gtr, Zte):
    """GBM design model; training residuals cross-fitted over inner group folds, test prediction from the full training fit."""
    f, k = M._fold_ids(gtr)
    ptr = np.zeros(len(ytr))
    for j in range(k):
        a, b = f != j, f == j
        ptr[b] = S.gbm_fit_predict(Ztr[a], ytr[a], gtr[a], Ztr[b])
    return ptr, S.gbm_fit_predict(Ztr, ytr, gtr, Zte)


def cell(a):
    t, cut = a
    y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
    XB, XC = B0, np.hstack([B0, G])
    rows = []
    for kind in ("lin", "quad", "gbm"):
        rt = np.zeros(len(y)); pb = np.zeros(len(y)); pc = np.zeros(len(y)); pg = np.zeros(len(y))
        for c in np.unique(g):
            te = g == c; tr = ~te
            if kind == "gbm":
                ftr, fte = gbm_design(DES[tr], y[tr], g[tr], DES[te])
            else:
                ftr, fte = R.design_fit_predict(DES[tr], y[tr], DES[te], kind)
            rtr = y[tr] - ftr; rt[te] = y[te] - fte
            pb[te] = S.ridge_fit_predict(XB[tr], rtr, g[tr], XB[te]); pc[te] = S.ridge_fit_predict(XC[tr], rtr, g[tr], XC[te])
        rows.append(("design-residual:" + kind, rt, pb, pc))
    # saturated config-mean model (within-config residual)
    yc = R.center(y[:, None], g)[:, 0]; Xb = R.center(B0, g); Xc = np.hstack([Xb, R.center(G, g)])
    rows.append(("config-mean (within-config, centred)", yc, S.loco(Xb, yc, g, "ridge"), S.loco(Xc, yc, g, "ridge")))
    out = []
    for var, r, pb, pc in rows:
        st = M.CfgStats(r, {"b": pb, "c": pc}, g)
        W = st.draw(2000, np.random.default_rng(0)); bs = st.d_r2("b", "c", W)
        d = st.d_r2("b", "c", np.ones((1, st.nc)))[0]
        q = np.percentile(bs, [2.5, 97.5, 5, 95])
        out.append(dict(dataset=M.DATASET, target=t, cutoff=cut, variant=var, resid_sd=float(r.std()), r2_base=S.r2(r, pb), r2_comp=S.r2(r, pc),
                        delta=d, lo95=q[0], hi95=q[1], lo90=q[2], hi90=q[3], **L.classify(q[2], q[3], L.margin(t))))
    return out


def main():
    global RUNS
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); RUNS = L.get_runs()
    with Pool(a.workers) as p:
        res = [r for rs in p.map(cell, [(t, c) for t in M.PRIMARY for c in M.CUTS]) for r in rs]
    M.write_csv(f"{M.OUT3}/resid_results.csv", res)
    lines = []
    for t in M.PRIMARY:
        lines.append(f"[{M.DATASET}] {t}: dR2 of G (B0 -> B0+G) on residual target, 95% config-bootstrap CI")
        for var in dict.fromkeys(r["variant"] for r in res):
            lines.append(f"  {var:40s} " + " | ".join(f"c{c}: {r['delta']:+.3f} [{r['lo95']:+.2f},{r['hi95']:+.2f}]" for c in M.CUTS for r in res if r["target"] == t and r["variant"] == var and r["cutoff"] == c))
    open(f"{M.OUT3}/resid_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
