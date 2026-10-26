#!/usr/bin/env python
"""Frozen orchestration + analysis for the Fashion-MNIST confirmatory replication
(see /vol/bitbucket/lbastian/latent-dynamics/analysis_v2/confirm_fm/PREREG.txt). DO NOT MODIFY AFTER LAUNCH.
Imports (never modifies) step7_confirm / step6_lib / step5_lib / step4_lib / step3_predict_ext; compute_metrics.py is run unchanged for the feature caches.
Dataset mode is 'fashion' (ANALYSIS_RUNSET=orig: 3 design columns log10_n, log10_wd, label_noise); B1 = 6 curve + 5 design (adds log10_lr, is_cnn).
usage:
  python step10_confirm_fm.py --extract     # compute missing caches (compute_metrics.py) then run everything (writes confirm_fm/results/)
  python step10_confirm_fm.py               # analysis only
  python step10_confirm_fm.py --dry-run     # smoke test: 10 OLD Fashion configs act as the 'new' set, rest as 'old'; secondary (e) skipped; writes confirm_fm/dryrun/
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "--dataset" not in sys.argv:          # step4_lib reads sys.argv at import: select the Fashion run set (ANALYSIS_RUNSET=orig)
    sys.argv += ["--dataset", "fashion"]
import step7_confirm as K7
import argparse, json, subprocess
import numpy as np
import confirm_fm_gen as G1
Z, M, L, S, C, D = K7.Z, K7.M, K7.L, K7.S, K7.C, K7.D
assert Z.DATASET == "fashion" and C.RUNSET == "orig"
CUTS, TARGETS, NB, NB_SEC = K7.CUTS, K7.TARGETS, K7.NB, K7.NB_SEC
margin, holm, ci_row, sentence, design5 = K7.margin, K7.holm, K7.ci_row, K7.sentence, K7.design5
BASE = os.path.join(C.OUT_ROOT, "confirm_fm")
lg = lambda x: np.log(np.maximum(x, 1e-6))
lgo = lambda x: np.log(np.maximum(np.abs(x), 1e-8))
RNG0 = 0


def new_rids():
    return [G1.job_line(c, s).split("|")[0] for c in G1.configs() for s in G1.NEW_SEEDS]


def load_runs_from(rids):
    runs = []
    for rid in rids:
        info = C.run_info(rid); cfg = json.load(open(os.path.join(C.RUNS, rid, "config.json"))); d = C.load_cache(rid)
        if d is None:
            continue
        info["config_id"] = cfg["tag"]                       # cfm_cNN
        info["lr"] = float(cfg["lr"]); info["is_cnn"] = int(cfg["encoder_arch"] == "cnn")
        steps = d["step"]; gap = d["run/train_acc"] - d["run/test_acc"]
        runs.append(dict(info=info, d=d, step=steps, covered=False, onsetN=np.nan, preN=len(steps) - 1, topo=d["topo_mask"].astype(bool),
                         onset=np.nan, pre=len(steps) - 1, gap=gap, peak_acc=float(np.nanmax(d["run/test_acc"])), peak_step=0, cross_step=np.nan))
    return runs


def get(runs, cut, t):
    return K7.get(runs, cut, t)


def loco_fn(build, y, g):
    p = np.zeros(len(y))
    for c in np.unique(g):
        te = g == c; tr = ~te
        Xtr, Xte = build(tr, te)
        p[te] = S.ridge_fit_predict(Xtr, y[tr], g[tr], Xte)
    return p


def center(X, g):
    Xc = X.copy()
    for c in np.unique(g):
        m = g == c; Xc[m] = X[m] - X[m].mean(0)
    return Xc


# ------------------------------------------------------------------ non-latent blocks
def curve_block(r, cut):
    """CV18 (as step8_e): 6 curve quantities x {value at last snapshot <= sc/10, OLS slope vs log10(step+1) over last <=5 snapshots, std over snapshots <= sc}."""
    d = r["d"]; st = r["step"]; k = D.idx_le_mask(st, r["topo"], cut); sc = st[k]
    tl, te, tra, tea = d["run/train_loss"], d["run/test_loss"], d["run/train_acc"], d["run/test_acc"]
    qs = [lg(tl), lg(te), tra, tea, tra - tea, lg(te) - lg(tl)]
    ok = np.where(st <= sc)[0]; i10 = np.where(st <= sc / 10)[0]; i10 = i10[-1] if len(i10) else 0
    out = []
    for q in qs:
        v = q[ok]; x = np.log10(st[ok] + 1.0)
        last = slice(max(0, len(ok) - 5), len(ok))
        sl = np.polyfit(x[last], v[last], 1)[0] if (len(ok) - max(0, len(ok) - 5) >= 3 and np.ptp(x[last]) > 0) else 0.0
        out += [q[i10], sl, v.std()]
    return np.array(out)


_OPT = {}
OPT_KEYS = ("opt_theta_norm", "opt_dtheta_norm", "opt_grad_norm", "opt_lambda_max", "opt_eff_step_q")


def opt_raw(r):
    """per-snapshot raw optimiser diagnostics for run r (all snapshots, read once from the test-probe snapshot npz; NaN at step 0)."""
    rid = r["info"]["run_id"]
    if rid not in _OPT:
        st = r["step"]; out = np.full((len(st), 6), np.nan)
        for j, s in enumerate(st):
            if s > 1000: continue                       # only snapshots <= max cutoff are ever used
            z = np.load(Z.snap_files(rid, "test", s))
            for k in OPT_KEYS:
                if k not in z.files:
                    raise KeyError(f"{rid}: {k} missing in snapshot {s}")
            q = np.asarray(z["opt_eff_step_q"], float)                                 # quantiles 1,10,50,90,99 %
            out[j] = [lgo(z["opt_theta_norm"]), lgo(z["opt_dtheta_norm"]), lgo(z["opt_grad_norm"]), lgo(z["opt_lambda_max"]), lgo(q[2]),
                      np.log(max(q[3], 1e-30)) - np.log(max(q[1], 1e-30))]
        _OPT[rid] = out
    return _OPT[rid]


def opt_blocks(r, cut):
    """OPT18 = 6 quantities (log theta norm, log update norm, log grad norm, log |lambda_max|, log median Adam effective step, log(q90/q10) of eff. step)
    x {value at the cutoff snapshot (same snapshot as G), value at last snapshot with 1<=step<=sc/10, OLS slope vs log10(step+1) over last <=5 snapshots with 1<=step<=sc}.
    OPT4 = value-at-cutoff of [log theta norm, log grad norm, log |lambda_max|, log median eff. step] (matches compact4 in size). Step 0 excluded (NaN update norm)."""
    R = opt_raw(r); st = r["step"]; k = D.idx_le_mask(st, r["topo"], cut); sc = st[k]
    ok = np.where((st >= 1) & (st <= sc))[0]
    if len(ok) == 0:
        ok = np.where(st >= 1)[0][:1]
    i10 = ok[st[ok] <= sc / 10]; i10 = i10[-1] if len(i10) else ok[0]
    lastv = ok[-1]; x = np.log10(st[ok] + 1.0); last = slice(max(0, len(ok) - 5), len(ok))
    o18 = []
    for q in range(6):
        v = R[ok, q]
        sl = np.polyfit(x[last], v[last], 1)[0] if (len(ok) - max(0, len(ok) - 5) >= 3 and np.ptp(x[last]) > 0) else 0.0
        o18 += [R[lastv, q], R[i10, q], sl]
    o4 = [R[lastv, 0], R[lastv, 2], R[lastv, 3], R[lastv, 4]]
    return np.nan_to_num(np.array(o18), nan=0.0), np.nan_to_num(np.array(o4), nan=0.0)


def pair_ci(st, A, B, W, W1, m):
    """mean over paired (a,b) keys (placebo draws) of the dR2 = R2(b)-R2(a), averaged INSIDE every bootstrap rep (step8_bcd 'C' convention)."""
    bs = np.mean([st.d_r2(a, b, W) for a, b in zip(A, B)], 0); ob = np.mean([st.d_r2(a, b, W1)[0] for a, b in zip(A, B)])
    q = np.percentile(bs, [2.5, 97.5, 5, 95]); nb = len(bs)
    return dict(delta=ob, lo95=q[0], hi95=q[1], lo90=q[2], hi90=q[3], p_pos=(np.sum(bs <= 0) + 1) / (nb + 1), p_neg=(np.sum(bs >= 0) + 1) / (nb + 1),
                tost_p=M.tost_p(bs, m), label=L.classify(q[2], q[3], m)["label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--extract", action="store_true"); ap.add_argument("--nb", type=int, default=NB)
    ap.add_argument("--dataset", default="fashion")
    a = ap.parse_args(); nb = a.nb
    out = os.path.join(BASE, "dryrun" if a.dry_run else "results"); os.makedirs(out, exist_ok=True)
    T = []
    log = lambda s="": (T.append(s), print(s, flush=True))
    # ------------------------------------------------------------------ data
    if a.dry_run:
        allold = L.load_fashion()
        cfgs = sorted({r["info"]["config_id"] for r in allold})[::3][:10]
        new = [r for r in allold if r["info"]["config_id"] in cfgs]
        for r in new:
            r["info"]["lr"] = 1e-3; r["info"]["is_cnn"] = 0
        old = [r for r in allold if r["info"]["config_id"] not in cfgs]
        log(f"DRY RUN: {len(new)} stand-in 'new' runs from {len(cfgs)} old Fashion configs; transfer trained on {len(old)} remaining old Fashion runs; secondary (e) skipped")
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
        old = L.load_fashion()
    nconf = len({r["info"]["config_id"] for r in new})
    log(f"new runs {len(new)}, new configs {nconf}, old Fashion runs {len(old)}")
    assert len(new) >= 6
    do_opt = not a.dry_run
    is_cnn = np.array([r["info"]["is_cnn"] for r in new], bool)
    # ------------------------------------------------------------------ primary (ridge) + (b) GBM + (c)
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
                W = st.draw(nb, np.random.default_rng(RNG0)); W1 = np.ones((1, st.nc))
                for k, X in (("F", "G"), ("C", "compact4")):
                    row = dict(model=mdl, target=t, cutoff=cut, X=X, margin=m, n_runs=len(y), n_cfg=st.nc, r2_B1=S.r2(y, P["b"]), r2_BX=S.r2(y, P[k]),
                               **ci_row(st, "b", k, W, m, W1))
                    (prim if mdl == "ridge" else sec).append(row)
            P = oof[(t, cut, "ridge")][2]
            for arch, mk in (("mlp", ~is_cnn), ("cnn", is_cnn)):
                if mk.sum() > 3:
                    perarch.append(dict(target=t, cutoff=cut, arch=arch, n_runs=int(mk.sum()), r2_B1=S.r2(y[mk], P["b"][mk]), r2_B1G=S.r2(y[mk], P["F"][mk]), r2_B1c4=S.r2(y[mk], P["C"][mk])))
    for X in ("G", "compact4"):
        fam = [r for r in prim if r["X"] == X]
        ph = holm([r["tost_p"] for r in fam]); pp = holm([r["p_pos"] for r in fam])
        for r, a1, a2 in zip(fam, ph, pp):
            r["tost_holm"] = a1; r["pos_holm"] = a2; r["equivalent_holm"] = int(a1 <= 0.05); r["positive_holm"] = int(a2 <= 0.05)
            r["possible_harm"] = int(r["lo90"] < -r["margin"]); r["sentence"] = sentence(r, "full G" if X == "G" else "compact4")
    M.write_csv(os.path.join(out, "primary_ridge.csv"), prim)
    M.write_csv(os.path.join(out, "secondary_b_gbm.csv"), sec)
    M.write_csv(os.path.join(out, "exploratory_per_arch.csv"), perarch)
    log("== PRIMARY (Fashion-MNIST; ridge LOCO over new configs; B1 vs B1+X; 90% CI; margins .03 acc / .05 gap; Holm over 10 cells per family) ==")
    for X in ("G", "compact4"):
        fam = [r for r in prim if r["X"] == X]
        log(f"-- family X={X}: HEADLINE (one-sided positive increment, Holm) {sum(r['positive_holm'] for r in fam)}/10 (unadjusted {sum(r['p_pos'] <= .05 for r in fam)}/10); "
            f"equivalence confirmed {sum(r['equivalent_holm'] for r in fam)}/10 (unadjusted {sum(r['tost_p'] <= .05 for r in fam)}/10)")
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
    # ------------------------------------------------------------------ (a) transfer: old Fashion -> new
    trows = []
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
                    W = st.draw(NB_SEC, np.random.default_rng(RNG0)); W1 = np.ones((1, st.nc))
                    for k, X in (("F", "G"), ("C", "compact4")):
                        trows.append(dict(model=mdl, subset=subset, target=t, cutoff=cut, X=X, n_runs=int(mk.sum()), n_cfg=st.nc, r2_B1=S.r2(yn[mk], P["b"][mk]),
                                          r2_BX=S.r2(yn[mk], P[k][mk]), rmse_B1_pts=100 * float(np.sqrt(np.mean((yn[mk] - P["b"][mk]) ** 2))),
                                          rmse_BX_pts=100 * float(np.sqrt(np.mean((yn[mk] - P[k][mk]) ** 2))), margin=m, **ci_row(st, "b", k, W, m, W1)))
    M.write_csv(os.path.join(out, "secondary_a_transfer.csv"), trows)
    log("== (a) transfer: fit on the old Fashion runs, test on new runs (descriptive) ==")
    for r in trows:
        log(f"   {r['model']:5s} {r['subset']:8s} {r['X']:8s} {r['target']:8s} cut{r['cutoff']:5d} R2(B1) {r['r2_B1']:+.3f} R2(B1+X) {r['r2_BX']:+.3f} RMSE pts {r['rmse_B1_pts']:.2f}->{r['rmse_BX_pts']:.2f} dR2 {r['delta']:+.3f} 90%[{r['lo90']:+.3f},{r['hi90']:+.3f}] {r['label']}")
    # ------------------------------------------------------------------ (d) placebos RP18 / NZ18 / CV18, (e) OPT18 / OPT4, (f) 1-dim blocks
    WS = Z.make_dirs(); nd = Z.NDRAW
    drows, erows, frows = [], [], []
    for cut in CUTS:
        Pf = np.zeros((len(new), nd, 18))
        for i, r in enumerate(new):
            st_ = r["step"]; k = D.idx_le_mask(st_, r["topo"], cut); rid = r["info"]["run_id"]; f = {}
            for ps in ("test", "train"):
                x = np.load(Z.snap_files(rid, ps, st_[k]))["latents"].astype(np.float64)
                f[ps] = np.stack([Z.proj_stats(x, W_) for W_ in WS])
            Pf[i] = np.hstack([f["test"], f["train"], f["train"] - f["test"]])
        CV = np.array([curve_block(r, cut) for r in new])
        if do_opt:
            OB = [opt_blocks(r, cut) for r in new]; O18 = np.array([o[0] for o in OB]); O4 = np.array([o[1] for o in OB])
        for t in TARGETS:
            m = margin(t)
            y, B0, DES3, Gm, g, names = get(new, cut, t)
            B1 = np.hstack([B0, design5(new, DES3)]); c4 = M.gvariant_cols(names[2], "compact4"); gn = names[2]; ix = gn.index
            P = {"b": oof[(t, cut, "ridge")][2]["b"], "F": oof[(t, cut, "ridge")][2]["F"], "C": oof[(t, cut, "ridge")][2]["C"]}
            for d_ in range(nd):
                P[f"rp{d_}"] = S.loco(np.hstack([B1, Pf[:, d_, :]]), y, g, "ridge"); P[f"nz{d_}"] = S.loco(np.hstack([B1, Z.noise_block(len(new), 18, d_)]), y, g, "ridge")
            P["cv"] = S.loco(np.hstack([B1, CV]), y, g, "ridge")
            if do_opt:
                P["o18"] = S.loco(np.hstack([B1, O18]), y, g, "ridge"); P["o4"] = S.loco(np.hstack([B1, O4]), y, g, "ridge")
            # (f) 1-dim blocks
            mk_cols = lambda cols: (lambda tr, te: (np.hstack([B1[tr], Gm[tr][:, cols]]), np.hstack([B1[te], Gm[te][:, cols]])))
            def mk_pca(tr, te):
                a_, b_ = M.pca_block(Gm[tr], Gm[te], 1); return np.hstack([B1[tr], a_]), np.hstack([B1[te], b_])
            f1 = {"te_fisher": mk_cols([ix("te_fisher")]), "te_eff_rank_pr": mk_cols([ix("te_eff_rank_pr")]), "te_mag_t1": mk_cols([ix("te_mag_t1")]), "PCA-1(G, in-fold)": mk_pca}
            for nm, bf in f1.items():
                P["f1_" + nm] = loco_fn(bf, y, g)
            st = M.CfgStats(y, P, g); W = st.draw(nb, np.random.default_rng(RNG0)); W1 = np.ones((1, st.nc))
            rp = [f"rp{d_}" for d_ in range(nd)]; nz = [f"nz{d_}" for d_ in range(nd)]
            base = dict(target=t, cutoff=cut, margin=m, r2_B1=S.r2(y, P["b"]))
            for nm, A, B in (("RP18 vs B1", ["b"] * nd, rp), ("NZ18 vs B1", ["b"] * nd, nz), ("CV18 vs B1", ["b"], ["cv"]),
                             ("G minus RP18", rp, ["F"] * nd), ("G minus NZ18", nz, ["F"] * nd), ("G minus CV18", ["cv"], ["F"]),
                             ("compact4 minus RP18", rp, ["C"] * nd), ("compact4 minus CV18", ["cv"], ["C"])):
                drows.append(dict(contrast=nm, **base, **pair_ci(st, A, B, W, W1, m)))
            if do_opt:
                for nm, A, B in (("OPT18 vs B1", ["b"], ["o18"]), ("OPT4 vs B1", ["b"], ["o4"]), ("G minus OPT18", ["o18"], ["F"]), ("compact4 minus OPT4", ["o4"], ["C"]),
                                 ("G minus OPT4", ["o4"], ["F"]), ("compact4 minus OPT18", ["o18"], ["C"])):
                    erows.append(dict(contrast=nm, **base, **pair_ci(st, A, B, W, W1, m)))
            for nm in f1:
                frows.append(dict(block=nm, **base, **pair_ci(st, ["b"], ["f1_" + nm], W, W1, m)))
            print("placebo/opt/1dim", t, cut, flush=True)
    for rows, key in ((drows, "contrast"), (erows, "contrast"), (frows, "block")):
        for nm in {r[key] for r in rows}:
            fam = [r for r in rows if r[key] == nm]
            for r, h in zip(fam, holm([r["p_pos"] for r in fam])): r["p_pos_holm"] = h
            for r, h in zip(fam, holm([r["p_neg"] for r in fam])): r["p_neg_holm"] = h
    M.write_csv(os.path.join(out, "secondary_d_placebo.csv"), drows)
    if erows: M.write_csv(os.path.join(out, "secondary_e_opt.csv"), erows)
    M.write_csv(os.path.join(out, "secondary_f_lowdim.csv"), frows)

    def show(rows, key, title):
        log(title)
        for r in rows:
            log(f"   {r[key]:22s} {r['target']:8s} cut{r['cutoff']:5d} dR2 {r['delta']:+.3f} 90%[{r['lo90']:+.3f},{r['hi90']:+.3f}] 95%[{r['lo95']:+.3f},{r['hi95']:+.3f}] "
                f"p+ {r['p_pos']:.4f} Holm {r['p_pos_holm']:.4f} | p- {r['p_neg']:.4f} Holm {r['p_neg_holm']:.4f}")
    show(drows, "contrast", "== (d) placebos (ridge LOCO; RP18/NZ18 = mean over 5 draws inside each bootstrap rep; 'X minus P' = R2(X)-R2(P); Holm over 10 cells per contrast; p+ = H0 d<=0, p- = H0 d>=0) ==")
    if erows: show(erows, "contrast", "== (e) non-geometric optimiser/weight block OPT18 / OPT4 (G minus OPT contrasts; Holm over 10 cells per contrast) ==")
    else: log("== (e) skipped (dry run: no opt_* keys) ==")
    show(frows, "block", "== (f) 1-dim blocks (B1 vs B1+block; Holm over 10 cells per block) ==")
    # ------------------------------------------------------------------ (g) within-config centred analysis (as step9_b_within)
    wrows = []
    for t in TARGETS:
        for cut in CUTS:
            y, B0, DES3, Gm, g, names = get(new, cut, t)
            yc = center(y[:, None], g)[:, 0]; Bc, Gc = center(B0, g), center(Gm, g); c4 = M.gvariant_cols(names[2], "compact4")
            PW = {"B0c": S.loco(Bc, yc, g, "ridge"), "B0c+G": S.loco(np.hstack([Bc, Gc]), yc, g, "ridge"), "B0c+c4": S.loco(np.hstack([Bc, Gc[:, c4]]), yc, g, "ridge"), "null": np.zeros(len(y))}
            if do_opt:
                O18c = center(np.array([opt_blocks(r, cut)[0] for r in new]), g)
                PW["B0c+OPT18"] = S.loco(np.hstack([Bc, O18c]), yc, g, "ridge")
            for d_ in range(nd):
                PW[f"nz{d_}"] = S.loco(np.hstack([Bc, center(Z.noise_block(len(y), 18, d_), g)]), yc, g, "ridge")
            st = M.CfgStats(yc, PW, g); W1 = np.ones(st.nc); W = st.draw(NB, np.random.default_rng(RNG0)); n1, nW = st.n.sum(), W @ st.n
            rm = lambda a_, w, nn: 100 * np.sqrt((w @ st.sse[a_]) / nn)
            sd_within = 100 * np.sqrt(np.mean([np.var(y[g == c_], ddof=1) for c_ in np.unique(g)]))
            base = dict(target=t, cutoff=cut, n_runs=len(y), n_cfg=st.nc, within_sd_pts=sd_within, null_rmse_pts=rm("null", W1, n1), B0c_rmse_pts=rm("B0c", W1, n1))
            for k in ["B0c+G", "B0c+c4"] + (["B0c+OPT18"] if do_opt else []) + ["nz"]:
                ks = [k] if k != "nz" else [f"nz{d_}" for d_ in range(nd)]
                dR = np.mean([(W1 @ st.sse["B0c"] - W1 @ st.sse[a_]) / (W1 @ st.sse["null"]) for a_ in ks])
                bR = np.mean([(W @ st.sse["B0c"] - W @ st.sse[a_]) / (W @ st.sse["null"]) for a_ in ks], 0)
                gain = np.mean([rm("B0c", W1, n1) - rm(a_, W1, n1) for a_ in ks])
                q = np.percentile(bR, [2.5, 97.5, 5, 95])
                wrows.append(dict(base, model=k if k != "nz" else "B0c+NZ18(centred, mean 5 draws)", dR2c=dR, dR2c_lo95=q[0], dR2c_hi95=q[1], dR2c_lo90=q[2], dR2c_hi90=q[3],
                                  rmse_gain_pts=gain, p_pos=(np.sum(bR <= 0) + 1) / (len(bR) + 1)))
            print("within", t, cut, flush=True)
    for mdl in {r["model"] for r in wrows}:
        fam = [r for r in wrows if r["model"] == mdl]
        for r, h in zip(fam, holm([r["p_pos"] for r in fam])): r["p_pos_holm"] = h
    M.write_csv(os.path.join(out, "secondary_g_within.csv"), wrows)
    log("== (g) within-config centred analysis (target and B0/G/OPT18 centred per config; dR2c relative to B0c; 95% config-bootstrap CI; one-sided p+ with Holm over 10 cells) ==")
    for r in wrows:
        log(f"   {r['model']:34s} {r['target']:8s} cut{r['cutoff']:5d} within-SD {r['within_sd_pts']:.2f} B0c RMSE {r['B0c_rmse_pts']:.2f} dR2c {r['dR2c']:+.3f} [{r['dR2c_lo95']:+.3f},{r['dR2c_hi95']:+.3f}] gain {r['rmse_gain_pts']:+.3f} pts p+ {r['p_pos']:.4f} Holm {r['p_pos_holm']:.4f}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(T)); print("wrote", out)


if __name__ == "__main__":
    main()
