#!/usr/bin/env python
"""Frozen confirmatory analysis (see /vol/bitbucket/lbastian/latent-dynamics/analysis_v2/confirm/PREREG.txt). DO NOT MODIFY AFTER LAUNCH.
Reuses step3_predict_ext / step4_lib / step5_lib / step5_main.holm / step6_lib unchanged; compute_metrics.py is run unchanged for feature caches.
usage:
  python step7_confirm.py --extract          # (once all runs are trained) computes missing caches with compute_metrics.py, then analyses
  python step7_confirm.py                    # analysis only (caches must exist)
  python step7_confirm.py --dry-run          # smoke test: treats 10 OLD Adam-20k configs as the 'new' set; writes to confirm/dryrun/
"""
import sys, os
import step6_lib as Z           # caps BLAS threads at 2, ANALYSIS_RUNSET=ext, dataset mnist
import step5_main as P5
import argparse, json, subprocess
import numpy as np
M, L, S, C, D = Z.M, Z.L, Z.S, Z.C, Z.D
import confirm_gen as G0

CUTS = [10, 30, 100, 300, 1000]
TARGETS = ["acc_15kw", "gap_15kw"]
NB = 10000
NB_SEC = 10000
NB_PLC = 4000
MARG = {"acc": 0.03, "gap": 0.05}
CONF = os.path.join(C.OUT_ROOT, "confirm")


def margin(t):
    return MARG[t[:3]]


def new_rids():
    return [G0.job_line(c, s).split("|")[0] for c in G0.configs() for s in G0.NEW_SEEDS]


def load_runs_from(rids, conf_ids=True):
    runs = []
    for rid in rids:
        info = C.run_info(rid); cfg = json.load(open(os.path.join(C.RUNS, rid, "config.json"))); d = C.load_cache(rid)
        if d is None:
            continue
        if conf_ids:
            info["config_id"] = "conf_" + cfg["tag"].split("conf_")[1]
        info["lr"] = float(cfg["lr"]); info["is_cnn"] = int(cfg["encoder_arch"] == "cnn")
        steps = d["step"]; gap = d["run/train_acc"] - d["run/test_acc"]
        runs.append(dict(info=info, d=d, step=steps, covered=False, onsetN=np.nan, preN=len(steps) - 1, topo=d["topo_mask"].astype(bool),
                         onset=np.nan, pre=len(steps) - 1, gap=gap, peak_acc=float(np.nanmax(d["run/test_acc"])), peak_step=0, cross_step=np.nan))
    return runs


def design5(runs, DES3):
    lr = np.array([np.log10(r["info"]["lr"]) for r in runs]); cnn = np.array([r["info"]["is_cnn"] for r in runs], float)
    return np.column_stack([DES3[:, :3], lr, cnn])        # log10_n, log10_wd, label_noise, log10_lr, is_cnn


def get(runs, cut, t):
    y = np.array([S.targets(r)[t] for r in runs])
    B0, DES, Gm, g, seeds, names = S.build(runs, cut)
    return y, B0, DES, Gm, g, names


def holm(p):
    return P5.holm(np.asarray(p, float))


def ci_row(st, a, b, W, m, W1):
    d = st.d_r2(a, b, W); obs = st.d_r2(a, b, W1)[0]
    q = np.percentile(d, [2.5, 97.5, 5, 95])
    return dict(delta=obs, lo95=q[0], hi95=q[1], lo90=q[2], hi90=q[3], tost_p=M.tost_p(d, m), p_pos=(np.sum(d <= 0) + 1) / (len(d) + 1),
                label=L.classify(q[2], q[3], m)["label"])


def sentence(r, X):
    t, c, m = r["target"], r["cutoff"], r["margin"]
    f = lambda v: f"{v:+.3f}"
    out = []
    if r["equivalent_holm"]:
        out.append(f"On the new, independently drawn configs, adding {X} to the baseline-curve+design features was statistically equivalent (Holm-adjusted TOST p = {r['tost_holm']:.4f}, "
                   f"90% CI [{f(r['lo90'])},{f(r['hi90'])}] for the change in R^2, margin +-{m}) at cutoff {c} for {t}.")
    else:
        out.append(f"On the new configs, equivalence of B1 and B1+{X} at cutoff {c} for {t} was not confirmed at the pre-registered margin +-{m} "
                   f"(90% CI [{f(r['lo90'])},{f(r['hi90'])}], Holm-adjusted TOST p = {r['tost_holm']:.4f}); this does not establish an increment.")
    if r["positive_holm"]:
        out.append(f"On the new configs, adding {X} produced a positive increment in R^2 at cutoff {c} for {t} (delta R^2 = {f(r['delta'])}, 90% CI [{f(r['lo90'])},{f(r['hi90'])}], Holm-adjusted one-sided p = {r['pos_holm']:.4f}).")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--extract", action="store_true"); ap.add_argument("--nb", type=int, default=NB)
    a = ap.parse_args()
    nb = a.nb
    out = os.path.join(CONF, "dryrun" if a.dry_run else "results"); os.makedirs(out, exist_ok=True)
    T = []
    log = lambda s="": (T.append(s), print(s, flush=True))
    # ------------------------------------------------------------------ data
    if a.dry_run:
        allold = D.load_runs("mnist")
        adam = [r for r in allold if r["info"]["block"] in ("E2", "A") and r["info"]["optimizer"] == "adam" and r["info"]["train_steps"] == 20000]
        cfgs = sorted({r["info"]["config_id"] for r in adam})[:10]
        new = [r for r in allold if r["info"]["config_id"] in cfgs]
        for r in new:
            r["info"]["lr"] = 1e-3; r["info"]["is_cnn"] = 0
        old = [r for r in allold if r["info"]["config_id"] not in cfgs]
        log(f"DRY RUN: {len(new)} stand-in 'new' runs from {len(cfgs)} old configs; transfer trained on {len(old)} remaining old runs")
    else:
        rids = new_rids()
        ok = [r for r in rids if os.path.exists(os.path.join(C.RUNS, r, "model_final.pt")) and os.path.isdir(os.path.join(C.RUNS, r, "latents_train"))]
        miss = sorted(set(rids) - set(ok))
        if a.extract:
            todo = [r for r in ok if not os.path.exists(os.path.join(C.CACHE, r + ".npz"))]
            if todo:
                env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
                subprocess.run([sys.executable, os.path.join(C.REPO, "analysis_v2", "compute_metrics.py"), "--workers", "6", "--runs", *todo], check=True, env=env)
        new = load_runs_from(ok)
        cnt = {}
        for r in new: cnt[r["info"]["config_id"]] = cnt.get(r["info"]["config_id"], 0) + 1
        drop = {c for c, k in cnt.items() if k < 2}
        new = [r for r in new if r["info"]["config_id"] not in drop]
        log(f"runs missing/unfinished or without cache: {len(rids) - len(new)} of {len(rids)}; dropped configs (<2 seeds): {sorted(drop)}; missing: {miss[:10]}{'...' if len(miss) > 10 else ''}")
        old = D.load_runs("mnist")
    log(f"new runs {len(new)}, new configs {len({r['info']['config_id'] for r in new})}, old runs {len(old)}")
    assert len(new) >= 6
    rng_seed = 0
    ones = None
    # ------------------------------------------------------------------ primary + (b) GBM + (c)
    prim, sec, perarch = [], [], []
    oof = {}
    for t in TARGETS:
        m = margin(t)
        for cut in CUTS:
            y, B0, DES3, Gm, g, names = get(new, cut, t)
            B1 = np.hstack([B0, design5(new, DES3)]); c4 = M.gvariant_cols(names[2], "compact4")
            sets = {"b": B1, "F": np.hstack([B1, Gm]), "C": np.hstack([B1, Gm[:, c4]])}
            for mdl in ("ridge", "gbm"):
                P = {k: S.loco(X, y, g, mdl) for k, X in sets.items()}
                oof[(t, cut, mdl)] = (y, g, P)
                st = M.CfgStats(y, P, g)
                W = st.draw(nb, np.random.default_rng(rng_seed)); W1 = np.ones((1, st.nc))
                for k, X in (("F", "G"), ("C", "compact4")):
                    row = dict(model=mdl, target=t, cutoff=cut, X=X, margin=m, n_runs=len(y), n_cfg=st.nc, r2_B1=S.r2(y, P["b"]), r2_BX=S.r2(y, P[k]),
                               **ci_row(st, "b", k, W, m, W1))
                    (prim if mdl == "ridge" else sec).append(row)
            # exploratory per-arch pooled LOCO R2
            is_cnn = np.array([r["info"]["is_cnn"] for r in new], bool)
            P = oof[(t, cut, "ridge")][2]
            for arch, mk in (("mlp", ~is_cnn), ("cnn", is_cnn)):
                if mk.sum() > 3:
                    perarch.append(dict(target=t, cutoff=cut, arch=arch, n_runs=int(mk.sum()), r2_B1=S.r2(y[mk], P["b"][mk]), r2_B1G=S.r2(y[mk], P["F"][mk]), r2_B1c4=S.r2(y[mk], P["C"][mk])))
    # Holm per family (ridge only), per X
    for X in ("G", "compact4"):
        fam = [r for r in prim if r["X"] == X]
        ph = holm([r["tost_p"] for r in fam]); pp = holm([r["p_pos"] for r in fam])
        for r, a1, a2 in zip(fam, ph, pp):
            r["tost_holm"] = a1; r["pos_holm"] = a2; r["equivalent_holm"] = int(a1 <= 0.05); r["positive_holm"] = int(a2 <= 0.05)
            r["possible_harm"] = int(r["lo90"] < -r["margin"]); r["sentence"] = sentence(r, "full G" if X == "G" else "compact4")
    M.write_csv(os.path.join(out, "primary_ridge.csv"), prim)
    M.write_csv(os.path.join(out, "secondary_b_gbm.csv"), sec)
    M.write_csv(os.path.join(out, "exploratory_per_arch.csv"), perarch)
    log("== PRIMARY (ridge LOCO over new configs; B1 vs B1+X; 90% CI; margins .03 acc / .05 gap; Holm over 10 cells per family) ==")
    for X in ("G", "compact4"):
        fam = [r for r in prim if r["X"] == X]
        log(f"-- family X={X}: equivalence confirmed {sum(r['equivalent_holm'] for r in fam)}/10 (unadjusted {sum(r['tost_p'] <= .05 for r in fam)}/10); positive increment {sum(r['positive_holm'] for r in fam)}/10 (unadjusted {sum(r['p_pos'] <= .05 for r in fam)}/10)")
        for r in fam:
            log(f"   {r['target']:8s} cut{r['cutoff']:5d} R2(B1) {r['r2_B1']:.3f} dR2 {r['delta']:+.3f} 90%[{r['lo90']:+.3f},{r['hi90']:+.3f}] TOSTp {r['tost_p']:.4f} Holm {r['tost_holm']:.4f} | p+ {r['p_pos']:.4f} Holm {r['pos_holm']:.4f} | "
                f"{'EQUIV-CONFIRMED' if r['equivalent_holm'] else 'not confirmed'}{' POSITIVE' if r['positive_holm'] else ''}{' possible-harm' if r['possible_harm'] else ''}")
    log("== SENTENCES (family F) ==")
    for r in [r for r in prim if r["X"] == "G"]: log(r["sentence"])
    log("== SENTENCES (family C) ==")
    for r in [r for r in prim if r["X"] == "compact4"]: log(r["sentence"])
    log("== (b) GBM LOCO, new configs (descriptive; no Holm) ==")
    for r in sec:
        log(f"   {r['X']:8s} {r['target']:8s} cut{r['cutoff']:5d} R2(B1) {r['r2_B1']:.3f} dR2 {r['delta']:+.3f} 90%[{r['lo90']:+.3f},{r['hi90']:+.3f}] 95%[{r['lo95']:+.3f},{r['hi95']:+.3f}] p+ {r['p_pos']:.4f} {r['label']}")
    log("== (c) gap_15kw cutoff 1000 ==")
    for r in prim + sec:
        if r["target"] == "gap_15kw" and r["cutoff"] == 1000:
            log(f"   {r['model']:6s} {r['X']:8s} dR2 {r['delta']:+.3f} 90%[{r['lo90']:+.3f},{r['hi90']:+.3f}] 95%[{r['lo95']:+.3f},{r['hi95']:+.3f}] one-sided p+ {r['p_pos']:.4f} {r['label']}")
    log("== exploratory: pooled ridge LOCO R2 by architecture ==")
    for r in perarch: log(f"   {r['target']:8s} cut{r['cutoff']:5d} {r['arch']} n={r['n_runs']} R2 B1 {r['r2_B1']:.3f} B1+G {r['r2_B1G']:.3f} B1+c4 {r['r2_B1c4']:.3f}")
    # ------------------------------------------------------------------ (a) transfer: old -> new
    trows = []
    is_cnn = np.array([r["info"]["is_cnn"] for r in new], bool)
    for t in TARGETS:
        m = margin(t)
        for cut in CUTS:
            yo, B0o, DESo, Go, go, nameso = get(old, cut, t)
            yn, B0n, DESn, Gn, gn, namesn = get(new, cut, t)
            c4 = M.gvariant_cols(nameso[2], "compact4")
            Xo = {"b": np.hstack([B0o, DESo]), "F": np.hstack([B0o, DESo, Go]), "C": np.hstack([B0o, DESo, Go[:, c4]])}
            Xn = {"b": np.hstack([B0n, DESn]), "F": np.hstack([B0n, DESn, Gn]), "C": np.hstack([B0n, DESn, Gn[:, c4]])}
            for mdl in ("ridge", "gbm"):
                fp = S.ridge_fit_predict if mdl == "ridge" else S.gbm_fit_predict
                P = {k: fp(Xo[k], yo, go, Xn[k]) for k in Xo}
                for subset, mk in (("all", np.ones(len(yn), bool)), ("mlp_only", ~is_cnn)):
                    if mk.sum() < 6: continue
                    st = M.CfgStats(yn[mk], {k: v[mk] for k, v in P.items()}, gn[mk])
                    W = st.draw(NB_SEC, np.random.default_rng(rng_seed)); W1 = np.ones((1, st.nc))
                    for k, X in (("F", "G"), ("C", "compact4")):
                        trows.append(dict(model=mdl, subset=subset, target=t, cutoff=cut, X=X, n_runs=int(mk.sum()), n_cfg=st.nc, r2_B1=S.r2(yn[mk], P["b"][mk]),
                                          r2_BX=S.r2(yn[mk], P[k][mk]), rmse_B1_pts=100 * float(np.sqrt(np.mean((yn[mk] - P["b"][mk]) ** 2))),
                                          rmse_BX_pts=100 * float(np.sqrt(np.mean((yn[mk] - P[k][mk]) ** 2))), margin=m, **ci_row(st, "b", k, W, m, W1)))
    M.write_csv(os.path.join(out, "secondary_a_transfer.csv"), trows)
    log("== (a) transfer: fit on old set (ii) runs, test on new runs (descriptive) ==")
    for r in trows:
        log(f"   {r['model']:5s} {r['subset']:8s} {r['X']:8s} {r['target']:8s} cut{r['cutoff']:5d} R2(B1) {r['r2_B1']:+.3f} R2(B1+X) {r['r2_BX']:+.3f} RMSE pts {r['rmse_B1_pts']:.2f}->{r['rmse_BX_pts']:.2f} dR2 {r['delta']:+.3f} 90%[{r['lo90']:+.3f},{r['hi90']:+.3f}] {r['label']}")
    # ------------------------------------------------------------------ (d) placebo random-projection / noise blocks
    WS = Z.make_dirs(); nd = Z.NDRAW
    prows = []
    for cut in CUTS:
        Pf = np.zeros((len(new), nd, 18))
        for i, r in enumerate(new):
            st_ = r["step"]; k = D.idx_le_mask(st_, r["topo"], cut); rid = r["info"]["run_id"]; f = {}
            for ps in ("test", "train"):
                x = np.load(Z.snap_files(rid, ps, st_[k]))["latents"].astype(np.float64)
                f[ps] = np.stack([Z.proj_stats(x, W) for W in WS])
            Pf[i] = np.hstack([f["test"], f["train"], f["train"] - f["test"]])
        for t in TARGETS:
            y, B0, DES3, Gm, g, names = get(new, cut, t)
            B1 = np.hstack([B0, design5(new, DES3)]); pb = oof[(t, cut, "ridge")][2]["b"]; pF = oof[(t, cut, "ridge")][2]["F"]
            for kind in ("RP18", "NZ18"):
                ds, los, his, cs, cl, ch = [], [], [], [], [], []
                for d_ in range(nd):
                    blk = Pf[:, d_, :] if kind == "RP18" else Z.noise_block(len(new), 18, d_)
                    pp = S.loco(np.hstack([B1, blk]), y, g, "ridge")
                    st = M.CfgStats(y, {"b": pb, "F": pF, "p": pp}, g)
                    W = st.draw(NB_PLC, np.random.default_rng(rng_seed)); W1 = np.ones((1, st.nc))
                    r1 = ci_row(st, "b", "p", W, margin(t), W1); r2 = ci_row(st, "p", "F", W, margin(t), W1)
                    ds.append(r1["delta"]); los.append(r1["lo95"]); his.append(r1["hi95"]); cs.append(r2["delta"]); cl.append(r2["lo95"]); ch.append(r2["hi95"])
                prows.append(dict(target=t, cutoff=cut, block=kind, ndraw=nd, dR2_mean=np.mean(ds), lo95_mean=np.mean(los), hi95_mean=np.mean(his), dR2_min=min(ds), dR2_max=max(ds),
                                  G_minus_placebo=np.mean(cs), GvsP_lo95=np.mean(cl), GvsP_hi95=np.mean(ch)))
    M.write_csv(os.path.join(out, "secondary_d_placebo.csv"), prows)
    log("== (d) placebo blocks (ridge LOCO; B1 -> B1+block; mean over 5 draws; G-minus-placebo = R2(B1+G)-R2(B1+block)) ==")
    for r in prows:
        log(f"   {r['block']} {r['target']:8s} cut{r['cutoff']:5d} dR2 {r['dR2_mean']:+.3f} [{r['lo95_mean']:+.3f},{r['hi95_mean']:+.3f}] draws[{r['dR2_min']:+.3f},{r['dR2_max']:+.3f}] G-minus-placebo {r['G_minus_placebo']:+.3f} [{r['GvsP_lo95']:+.3f},{r['GvsP_hi95']:+.3f}]")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(T)); print("wrote", out)


if __name__ == "__main__":
    main()
