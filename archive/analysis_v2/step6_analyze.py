#!/usr/bin/env python
"""A5 analyses A (placebo), B (hard-case subsets), D (compact-geometry extrapolation tables) from OUT4/oof6.pkl. usage: --dataset mnist|fashion"""
import step6_lib as Z
import step4_splits as SP
import argparse, pickle
import numpy as np
L, S, M = Z.L, Z.S, Z.M
NB = 10000
CUTS = L.CUTS


def dr2(y, pa, pb, g, nb=NB, seed=0):
    ok = np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb, g = y[ok], pa[ok], pb[ok], g[ok]
    st = M.CfgStats(y, {"a": pa, "b": pb}, g)
    d = st.d_r2("a", "b", np.ones((1, st.nc)))[0]
    bs = st.d_r2("a", "b", st.draw(nb, np.random.default_rng(seed)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=d, lo95=lo, hi95=hi, r2_a=S.r2(y, pa), r2_b=S.r2(y, pb), n_runs=int(ok.sum()), n_cfg=st.nc)


def splits_of(res_cell, RUNS):
    names = ["loco"] + list(SP.split_defs(RUNS).keys())
    return [n for n in names if n in res_cell]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); a = ap.parse_args()
    RUNS = L.get_runs()
    res = pickle.load(open(f"{Z.OUT4}/oof6.pkl", "rb"))
    folds = {"loco": None}; folds.update({k: v for k, v in SP.split_defs(RUNS).items()})
    drawsA, sumA, rowsD, rowsA0 = [], [], [], []
    KINDS = [("Prp", "random-projection"), ("Pnz", "gaussian-noise")]
    SZ = [("18", "full G (18)", "B1+G"), ("8", "compact8", "B1+c8"), ("4", "compact4", "B1+c4")]
    for (t, cut, mdl), cell in sorted(res.items()):
        y, g = cell["y"], cell["g"]
        nd = Z.NDRAW if mdl == "ridge" else Z.NDRAW_GBM
        for split in splits_of(cell, RUNS):
            P = cell[split]
            base = dict(dataset=Z.DATASET, split=split, model=mdl, target=t, cutoff=cut)
            # ---- A: placebo vs G at matched size
            for sz, szname, gk in SZ:
                rg = dr2(y, P["B1"], P[gk], g)
                sumA.append(dict(base, base_set="B1", block="G", size=szname, ndraw=1, dR2=rg["delta"], lo95=rg["lo95"], hi95=rg["hi95"],
                                 dR2_min=rg["delta"], dR2_max=rg["delta"], n_hurt_draws=int(rg["hi95"] < 0), n_pos_draws=int(rg["lo95"] > 0),
                                 r2_base=rg["r2_a"], GvsP_delta=np.nan, GvsP_lo=np.nan, GvsP_hi=np.nan, n_runs=rg["n_runs"], n_cfg=rg["n_cfg"]))
                for kk, kname in KINDS:
                    ds, los, his, gl, glo, ghi = [], [], [], [], [], []
                    for d in range(nd):
                        pk = f"B1+{kk}{sz}_d{d}"
                        r = dr2(y, P["B1"], P[pk], g, nb=4000)
                        ds.append(r["delta"]); los.append(r["lo95"]); his.append(r["hi95"])
                        drawsA.append(dict(base, base_set="B1", block=kname, size=szname, draw=d, dR2=r["delta"], lo95=r["lo95"], hi95=r["hi95"]))
                        q = dr2(y, P[pk], P[gk], g, nb=4000)    # R2(G)-R2(P)
                        gl.append(q["delta"]); glo.append(q["lo95"]); ghi.append(q["hi95"])
                    sumA.append(dict(base, base_set="B1", block=kname, size=szname, ndraw=nd, dR2=np.mean(ds), lo95=np.mean(los), hi95=np.mean(his),
                                     dR2_min=min(ds), dR2_max=max(ds), n_hurt_draws=int(sum(h < 0 for h in his)), n_pos_draws=int(sum(l > 0 for l in los)),
                                     r2_base=rg["r2_a"], GvsP_delta=np.mean(gl), GvsP_lo=np.mean(glo), GvsP_hi=np.mean(ghi), n_runs=rg["n_runs"], n_cfg=rg["n_cfg"]))
            if "B1+Pnorm" in P:
                r = dr2(y, P["B1"], P["B1+Pnorm"], g)
                sumA.append(dict(base, base_set="B1", block="train-probe raw norm (2)", size="2", ndraw=1, dR2=r["delta"], lo95=r["lo95"], hi95=r["hi95"],
                                 r2_base=r["r2_a"], n_runs=r["n_runs"], n_cfg=r["n_cfg"]))
            # ---- A (B0): design-proxy at the early cutoffs, LOCO only
            if split == "loco":
                for sz, szname, gk in (("18", "full G (18)", "B0+G"), ("4", "compact4", "B0+c4")):
                    rg = dr2(y, P["B0"], P[gk], g)
                    sumA.append(dict(base, base_set="B0", block="G", size=szname, ndraw=1, dR2=rg["delta"], lo95=rg["lo95"], hi95=rg["hi95"],
                                     dR2_min=rg["delta"], dR2_max=rg["delta"], n_hurt_draws=int(rg["hi95"] < 0), n_pos_draws=int(rg["lo95"] > 0),
                                     r2_base=rg["r2_a"], n_runs=rg["n_runs"], n_cfg=rg["n_cfg"]))
                    for kk, kname in KINDS:
                        ds, los, his, gl, glo, ghi = [], [], [], [], [], []
                        for d in range(nd):
                            pk = f"B0+{kk}{sz}_d{d}"
                            r = dr2(y, P["B0"], P[pk], g, nb=4000); ds.append(r["delta"]); los.append(r["lo95"]); his.append(r["hi95"])
                            q = dr2(y, P[pk], P[gk], g, nb=4000); gl.append(q["delta"]); glo.append(q["lo95"]); ghi.append(q["hi95"])
                        sumA.append(dict(base, base_set="B0", block=kname, size=szname, ndraw=nd, dR2=np.mean(ds), lo95=np.mean(los), hi95=np.mean(his),
                                         dR2_min=min(ds), dR2_max=max(ds), n_hurt_draws=int(sum(h < 0 for h in his)), n_pos_draws=int(sum(l > 0 for l in los)),
                                         r2_base=rg["r2_a"], GvsP_delta=np.mean(gl), GvsP_lo=np.mean(glo), GvsP_hi=np.mean(ghi), n_runs=rg["n_runs"], n_cfg=rg["n_cfg"]))
            # ---- D: compact tables, pooled + per fold
            for gk, gname in (("B1+c4", "compact4"), ("B1+c8", "compact8"), ("B1+G", "full G")):
                r = dr2(y, P["B1"], P[gk], g)
                rowsD.append(dict(base, fold="POOLED", variant=gname, dR2=r["delta"], lo95=r["lo95"], hi95=r["hi95"], r2_B1=r["r2_a"], r2_var=r["r2_b"],
                                  n_runs=r["n_runs"], n_cfg=r["n_cfg"], ci_excludes0=int(r["lo95"] > 0 or r["hi95"] < 0), hurts=int(r["hi95"] < 0)))
                if split != "loco" and folds[split] and (len(folds[split]) > 1 or split == "leave_one_optimizer_out"):
                    for nm, m in folds[split]:
                        r = dr2(y[m], P["B1"][m], P[gk][m], g[m], nb=4000)
                        rowsD.append(dict(base, fold=nm, variant=gname, dR2=r["delta"], lo95=r["lo95"], hi95=r["hi95"], r2_B1=r["r2_a"], r2_var=r["r2_b"],
                                          n_runs=r["n_runs"], n_cfg=r["n_cfg"], ci_excludes0=int(r["lo95"] > 0 or r["hi95"] < 0), hurts=int(r["hi95"] < 0)))
        print(t, cut, mdl, flush=True)
    M.write_csv(f"{Z.OUT4}/A_placebo_draws.csv", drawsA)
    M.write_csv(f"{Z.OUT4}/A_placebo_summary.csv", sumA)
    M.write_csv(f"{Z.OUT4}/D_compact_extrapolation.csv", rowsD)
    # ---------------- text tables
    T = []
    for mdl in ("ridge", "gbm"):
        for bs in ("B1", "B0"):
            for split in ["loco"] + [k for k in folds if k != "loco"]:
                for t in M.PRIMARY:
                    sub = [r for r in sumA if r["model"] == mdl and r["split"] == split and r["target"] == t and r["base_set"] == bs]
                    if not sub: continue
                    T.append(f"== {Z.DATASET} {mdl} {split} {t}: {bs} -> {bs}+X   dR2 [95% CI] (placebo = mean over draws, mean CI bounds; [min..max] draws; h/p = draws with CI<0 / >0) ==")
                    T.append("  block / size".ljust(44) + "".join(f"cut{c:<28d}" for c in CUTS))
                    keys = list(dict.fromkeys((r["block"], r["size"]) for r in sub))
                    for k in keys:
                        line = f"  {k[0][:22]:22s} {k[1][:18]:18s}  "
                        for c in CUTS:
                            r = next((r for r in sub if r["block"] == k[0] and r["size"] == k[1] and r["cutoff"] == c), None)
                            if r is None: line += " " * 22; continue
                            extra = f" h{r['n_hurt_draws']}p{r['n_pos_draws']}" if r["ndraw"] > 1 else ""
                            line += f"{r['dR2']:+.3f}[{r['lo95']:+.2f},{r['hi95']:+.2f}]{extra}".ljust(31)
                        T.append(line)
    open(f"{Z.OUT4}/A_placebo_summary.txt", "w").write("\n".join(T))
    T = []
    for mdl in ("ridge", "gbm"):
        for split in [k for k in folds]:
            for t in M.PRIMARY:
                sub = [r for r in rowsD if r["model"] == mdl and r["split"] == split and r["target"] == t and r["fold"] == "POOLED"]
                if not sub: continue
                T.append(f"== {Z.DATASET} {mdl} {split} {t}: dR2 [95% CI] for B1 -> B1+X (pooled; R2_B1 in last row) ==")
                T.append("  variant".ljust(14) + "".join(f"cut{c:<19d}" for c in CUTS))
                for v in ("compact4", "compact8", "full G"):
                    line = f"  {v:12s}"
                    for c in CUTS:
                        r = next(r for r in sub if r["variant"] == v and r["cutoff"] == c)
                        line += f"{r['dR2']:+.3f}[{r['lo95']:+.2f},{r['hi95']:+.2f}]".ljust(23)
                    T.append(line)
                T.append("  R2_B1       " + "".join(f"{next(r for r in sub if r['variant']=='full G' and r['cutoff']==c)['r2_B1']:+.3f}".ljust(23) for c in CUTS))
    open(f"{Z.OUT4}/D_compact_extrapolation.txt", "w").write("\n".join(T))

    # ---------------- B: hard-case subsets (selected by outcome -> exploratory)
    info = {c: None for c in np.unique(res[(M.PRIMARY[0], CUTS[0], "ridge")]["g"])}
    gg = res[(M.PRIMARY[0], CUTS[0], "ridge")]["g"]
    for r in RUNS: info[r["info"]["config_id"]] = r["info"]
    rowsB = []; Tb = []
    cfgs = np.array(sorted(info))
    for t in M.PRIMARY:
        mse = []
        for cut in CUTS:
            c_ = res[(t, cut, "ridge")]; y, g, p = c_["y"], c_["g"], c_["loco"]["B1"]
            mse.append(np.array([np.mean((y[g == c] - p[g == c]) ** 2) for c in cfgs]))
        mse = np.mean(mse, 0); thr = np.quantile(mse, 2 / 3)
        hard = {c for c, m in zip(cfgs, mse) if m >= thr}
        subs = {"hard_tercile_B1err": lambda c: c in hard, "easy_rest": lambda c: c not in hard,
                "label_noise>0": lambda c: info[c]["label_noise"] > 0, "offgrid_wd(E2)": lambda c: info[c]["block"] == "E2",
                "no_label_noise": lambda c: info[c]["label_noise"] == 0}
        if Z.DATASET == "fashion": subs.pop("offgrid_wd(E2)")
        for sname, fn in subs.items():
            sel = {c for c in cfgs if fn(c)}
            Tb.append(f"{Z.DATASET} {t} subset {sname}: {len(sel)} configs" + (f" (threshold mean config MSE {thr:.2e}; composition: " + ", ".join(sorted(sel))[:300] + ")" if sname == "hard_tercile_B1err" else ""))
            for mdl in ("ridge", "gbm"):
                for cut in CUTS:
                    c_ = res[(t, cut, mdl)]; y, g, P = c_["y"], c_["g"], c_["loco"]
                    m = np.array([c in sel for c in g])
                    if len(sel) < 4: continue
                    for lab, key in (("G full", "B1+G"), ("compact4", "B1+c4"), ("compact8", "B1+c8"), ("placebo RP18 (mean 3 draws)", "RP"), ("placebo noise18 (mean 3 draws)", "NZ"),
                                     ("placebo RP4 (mean 3 draws)", "RP4"), ("placebo noise4 (mean 3 draws)", "NZ4")):
                        if key in ("RP", "NZ", "RP4", "NZ4"):
                            sz = "18" if key in ("RP", "NZ") else "4"; kk = "Prp" if key.startswith("RP") else "Pnz"
                            rs = [dr2(y[m], P["B1"][m], P[f"B1+{kk}{sz}_d{d}"][m], g[m], nb=4000) for d in range(3)]
                            r = dict(delta=np.mean([x["delta"] for x in rs]), lo95=np.mean([x["lo95"] for x in rs]), hi95=np.mean([x["hi95"] for x in rs]), r2_a=rs[0]["r2_a"], n_runs=rs[0]["n_runs"], n_cfg=rs[0]["n_cfg"])
                        else:
                            r = dr2(y[m], P["B1"][m], P[key][m], g[m])
                        rowsB.append(dict(dataset=Z.DATASET, target=t, subset=sname, n_cfg=r["n_cfg"], n_runs=r["n_runs"], model=mdl, cutoff=cut, mode="eval_on_subset(trained on all other configs)",
                                          variant=lab, r2_B1=r["r2_a"], dR2=r["delta"], lo95=r["lo95"], hi95=r["hi95"]))
        # refit-within-subset (ridge): LOCO inside the subset only
        for sname, fn in subs.items():
            sel = {c for c in cfgs if fn(c)}
            if len(sel) < 8: continue
            for cut in CUTS:
                y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
                m = np.array([c in sel for c in g]); B1 = np.hstack([B0, DES])
                c4 = M.gvariant_cols(names[2], "compact4")
                pb = S.loco(B1[m], y[m], g[m], "ridge")
                for lab, X in (("G full", np.hstack([B1, G])), ("compact4", np.hstack([B1, G[:, c4]]))):
                    pv = S.loco(X[m], y[m], g[m], "ridge"); r = dr2(y[m], pb, pv, g[m])
                    rowsB.append(dict(dataset=Z.DATASET, target=t, subset=sname, n_cfg=r["n_cfg"], n_runs=r["n_runs"], model="ridge", cutoff=cut, mode="refit_within_subset_LOCO",
                                      variant=lab, r2_B1=r["r2_a"], dR2=r["delta"], lo95=r["lo95"], hi95=r["hi95"]))
    M.write_csv(f"{Z.OUT4}/B_hardcase.csv", rowsB)
    for mode in ("eval_on_subset(trained on all other configs)", "refit_within_subset_LOCO"):
        for t in M.PRIMARY:
            for sname in dict.fromkeys(r["subset"] for r in rowsB):
                for mdl in ("ridge", "gbm"):
                    sub = [r for r in rowsB if r["mode"] == mode and r["target"] == t and r["subset"] == sname and r["model"] == mdl]
                    if not sub: continue
                    Tb.append(f"-- {Z.DATASET} {t} [{sname}, {sub[0]['n_cfg']} cfgs/{sub[0]['n_runs']} runs] {mdl} {mode}: dR2 [95% CI]")
                    Tb.append("   variant".ljust(40) + "".join(f"cut{c:<19d}" for c in CUTS))
                    for v in dict.fromkeys(r["variant"] for r in sub):
                        line = f"   {v:36s}"
                        for c in CUTS:
                            r = next(r for r in sub if r["variant"] == v and r["cutoff"] == c)
                            line += f"{r['dR2']:+.3f}[{r['lo95']:+.2f},{r['hi95']:+.2f}]".ljust(23)
                        Tb.append(line)
                    Tb.append("   R2_B1".ljust(40) + "".join(f"{next(r for r in sub if r['cutoff']==c)['r2_B1']:+.3f}".ljust(23) for c in CUTS))
    open(f"{Z.OUT4}/B_hardcase.txt", "w").write("EXPLORATORY: subsets are chosen by outcome (B1 error) or by design; selection by B1 error inflates any added block's apparent gain (regression to the mean), see placebo rows.\n" + "\n".join(Tb))
    print("done")


if __name__ == "__main__":
    main()
