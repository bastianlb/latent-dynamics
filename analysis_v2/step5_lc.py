#!/usr/bin/env python
"""Amendment 4 A4.7: learning-curve baselines (monotone-smoothed current value, slope) on gap/acc. usage: --dataset"""
import step5_lib as M
import argparse
from multiprocessing import Pool
import numpy as np
from sklearn.isotonic import IsotonicRegression
L, S = M.L, M.S
RUNS = None


def slope(x, v):
    if len(np.unique(x)) < 3: return 0.0
    return float(np.polyfit(x, v, 1)[0])


def lc_feats(r, cut, which):
    st = r["step"]; m = st <= cut
    if m.sum() == 0: m[0] = True
    curve = (r["gap"] if which == "gap" else r["d"]["run/test_acc"])[m]
    x = np.log10(st[m] + 1.0)
    iso = IsotonicRegression(increasing=True).fit_transform(np.arange(len(curve)), curve)[-1]
    med3 = float(np.median(curve[-3:]))
    sl = slope(x[-5:], curve[-5:])
    md = st[m] >= cut / 10
    sld = slope(x[md], curve[md]) if md.sum() >= 3 else 0.0
    return [iso, med3, sl, sld]


def cell(a):
    t, cut = a
    y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
    which = "gap" if t.startswith("gap") else "acc"
    F = np.array([lc_feats(r, cut, which) for r in RUNS])
    B1 = np.hstack([B0, DES])
    Xs = {"B1": B1, "+iso": np.hstack([B1, F[:, [0]]]), "+med3": np.hstack([B1, F[:, [1]]]), "+slope": np.hstack([B1, F[:, [2]]]),
          "+slope+slope_dec": np.hstack([B1, F[:, [2, 3]]]), "+iso+slope+slope_dec": np.hstack([B1, F[:, [0, 2, 3]]])}
    rows = []
    for mdl in ("ridge", "gbm"):
        P = {}
        for k, X in Xs.items():
            P[k] = S.loco(X, y, g, mdl); P[k + "+G"] = S.loco(np.hstack([X, G]), y, g, mdl)
        st = M.CfgStats(y, P, g); W = st.draw(2000, np.random.default_rng(0)); W1 = np.ones((1, st.nc))
        def add(kind, a_, b_):
            bs = st.d_r2(a_, b_, W); q = np.percentile(bs, [2.5, 97.5, 5, 95])
            rows.append(dict(dataset=M.DATASET, target=t, cutoff=cut, model=mdl, comparison=kind, base=a_, comp=b_, r2_base=S.r2(y, P[a_]), r2_comp=S.r2(y, P[b_]),
                             delta=st.d_r2(a_, b_, W1)[0], lo95=q[0], hi95=q[1], **L.classify(q[2], q[3], L.margin(t))))
        for k in list(Xs)[1:]:
            add("baseline_gain", "B1", k)
        for k in Xs:
            add("G_increment", k, k + "+G")
    return rows


def main():
    global RUNS
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); RUNS = L.get_runs()
    with Pool(a.workers) as p:
        res = [r for rs in p.map(cell, [(t, c) for t in M.PRIMARY for c in M.CUTS]) for r in rs]
    M.write_csv(f"{M.OUT3}/lc_results.csv", res)
    lines = []
    for t in M.PRIMARY:
        for mdl in ("ridge", "gbm"):
            for comp in ("baseline_gain", "G_increment"):
                lines.append(f"[{M.DATASET}] {t} {mdl} {comp}")
                for b in dict.fromkeys(r["comp"] if comp == "baseline_gain" else r["base"] for r in res if r["comparison"] == comp):
                    lines.append(f"   {b:24s} " + " | ".join(f"c{c}: {r['delta']:+.3f} [{r['lo95']:+.2f},{r['hi95']:+.2f}]" for c in M.CUTS for r in res
                                 if r["target"] == t and r["model"] == mdl and r["comparison"] == comp and r["cutoff"] == c and (r["comp"] if comp == "baseline_gain" else r["base"]) == b))
    open(f"{M.OUT3}/lc_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
