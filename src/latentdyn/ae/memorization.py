"""Memorization-in-latent-space analysis for AE (and matched classifier probe) runs.  CPU only, <=8 threads.

Usage: python ae_memorization.py [--runs GLOB[,GLOB]] [--out DIR]
Outputs OUT (default /vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/memorization): <run_id>.npz per run + results.json (aggregates)

Per run, per snapshot t (train probe = the n_tr training points; test probe = 1000 test points):
 * MATCHED-SIZE geometry: the test probe is subsampled (REPS times, label-stratified) to n_tr points, so effective rank / spread are not biased by n_tr=250 vs 1000.
   spread_lab / spread_km use the UNBIASED pooled within-group variance (SSW/(n-#groups)); var_ratio total variance (n-1); effrank = participation ratio of centred
   singular values^2; var_ratio_white = tr(S_tr S_te^-1)/d (S_te from the full test probe) -- invariant to any invertible linear re-coordinatisation (gauge-free).
   nn_ratio = median_{test} min_{train} ||z-z'|| / median_{train} min_{other train} ||z-z'||.
 * dgm_ratio: LOO distance of a point to its (label) group mean, train / test, each normalised by sqrt(1+1/(n_c-1)) and the common latent scale.
 * disp_ratio: median per-interval latent displacement train/test (normalised by the same global scale), snapshots are log-spaced so intervals vary.
 * per-sample recon_err (AE only): t_mem_i = first step where err_i < MEMFRAC * min_t mean(test recon_err) ("fits better than any test point could on average").
   features: label, ink (|x|^2), pixel kNN density within the train probe, early difficulty (err_i at the step of the test-recon minimum), latent LOO dist to class mean at early/late time,
   path length / net displacement of each train point (gauge-dependent in absolute terms -> normalised by global scale, compared only within the same run).
"""
import os
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, json, re
import numpy as np
from latentdyn.ae import analysis as A

ROOT = A.ROOT
OUT = f"{ROOT}/ae_dynamics/analysis/memorization"
REPS = 5
CACHE = f"{A.OUT}/_cache"  # overridden by --cache
MEMFRAC = 0.5


def pooled_within(Z, g):
    """unbiased pooled within-group variance trace: SSW/(n-#groups present)."""
    ss = 0.0; k = 0
    for c in np.unique(g):
        zc = Z[g == c]
        if len(zc) < 2: continue
        ss += ((zc - zc.mean(0)) ** 2).sum(); k += 1
    n = sum((g == c).sum() for c in np.unique(g) if (g == c).sum() >= 2)
    return ss / max(n - k, 1)


def totvar(Z): return ((Z - Z.mean(0)) ** 2).sum() / (len(Z) - 1)


def effrank1(Z):
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2; return s.sum() ** 2 / (s ** 2).sum()


def nn_med(X, Y, same=False):
    D = A.sqd(X, Y)
    if same: np.fill_diagonal(D, np.inf)
    return np.sqrt(np.median(D.min(1)))


def loo_dgm(Z, g):
    """LOO distance of each point to its group mean, corrected by sqrt(1+1/(n_c-1)) so E||.||^2 matches the population within-variance."""
    d = np.full(len(Z), np.nan)
    for c in np.unique(g):
        ii = np.where(g == c)[0]; n = len(ii)
        if n < 3: continue
        s = Z[ii].sum(0)
        for i in ii: d[i] = np.linalg.norm(Z[i] - (s - Z[i]) / (n - 1)) / np.sqrt(1 + 1 / (n - 1))
    return d


def spearman(a, b): return A.spearman(np.asarray(a, float), np.asarray(b, float))


def analyze(rd, out):
    cfg = json.load(open(f"{rd}/config.json")); rid = cfg["run_id"]; ds = cfg["dataset"]
    steps, Zte, yte, rec_te, sc = A.load_snaps(rd, "latents"); _, Ztr, ytr, rec_tr, _ = A.load_snaps(rd, "latents_train")
    T, nte, d = Zte.shape; ntr = Ztr.shape[1]; rs = np.random.RandomState(0)
    Xte_all, _ = A.images(ds, False); Xtr_all, _ = A.images(ds, True)
    pte = np.load(f"{rd}/probe_index.npz")["indices"]; ptr = np.load(f"{rd}/probe_train_index.npz")["indices"]
    Xte, Xtr = Xte_all[pte].astype(np.float64), Xtr_all[ptr].astype(np.float64)
    assign = A.group_model(ds, CACHE); gte = {"lab": yte, "km": assign(Xte)}; gtr = {"lab": ytr, "km": assign(Xtr)}
    # stratified (by label) subsample indices of the test probe
    subs = []
    for _ in range(REPS):
        idx = []
        for c in range(10):
            ic = np.where(yte == c)[0]; k = int(round(ntr * (ytr == c).mean())); idx += list(rs.choice(ic, min(k, len(ic)), replace=False))
        subs.append(np.array(sorted(idx)))
    R = dict(steps=steps, ntr=ntr)
    for k in ("spread_lab", "spread_km", "var_ratio", "effrank_ratio", "var_ratio_white", "nn_ratio", "nn_te2tr", "nn_tr2tr", "dgm_ratio", "disp_ratio", "scale_te"):
        R[k] = np.zeros(T)
    dgm_tr = np.zeros((T, ntr)); dgm_te = np.zeros((T, nte))
    for t in range(T):
        zt, ze = Ztr[t], Zte[t]; sl, sk, vr, er = [], [], [], []
        for ix in subs:
            sl.append(pooled_within(zt, ytr) / pooled_within(ze[ix], yte[ix])); sk.append(pooled_within(zt, gtr["km"]) / pooled_within(ze[ix], gte["km"][ix]))
            vr.append(totvar(zt) / totvar(ze[ix])); er.append(effrank1(zt) / effrank1(ze[ix]))
        R["spread_lab"][t], R["spread_km"][t], R["var_ratio"][t], R["effrank_ratio"][t] = map(np.mean, (sl, sk, vr, er))
        Ste = np.cov(ze.T) + 1e-9 * np.eye(d) * np.trace(np.cov(ze.T)) / d; Str = np.cov(zt.T)
        R["var_ratio_white"][t] = np.trace(Str @ np.linalg.inv(Ste)) / d
        a, b = nn_med(ze, zt), nn_med(zt, zt, True); R["nn_te2tr"][t] = a; R["nn_tr2tr"][t] = b; R["nn_ratio"][t] = a / (b + 1e-300)
        R["scale_te"][t] = np.sqrt(totvar(ze))
        dgm_tr[t] = loo_dgm(zt, ytr) / R["scale_te"][t]; dgm_te[t] = loo_dgm(ze, yte) / R["scale_te"][t]
    R["dgm_ratio"] = np.nanmean(dgm_tr, 1) / np.nanmean(dgm_te, 1)
    # displacement per interval (normalised by the global scale at the later snapshot)
    dtr = np.linalg.norm(np.diff(Ztr, axis=0), axis=2) / R["scale_te"][1:, None]; dte = np.linalg.norm(np.diff(Zte, axis=0), axis=2) / R["scale_te"][1:, None]
    R["disp_ratio"][1:] = np.median(dtr, 1) / np.median(dte, 1); R["disp_ratio"][0] = np.nan
    pl_tr = dtr.sum(0); pl_te = dte.sum(0); net_tr = np.linalg.norm(Ztr[-1] - Ztr[0], axis=1) / R["scale_te"].mean(); net_te = np.linalg.norm(Zte[-1] - Zte[0], axis=1) / R["scale_te"].mean()
    R["pathlen_tr"], R["pathlen_te"] = pl_tr, pl_te; R["net_tr"], R["net_te"] = net_tr, net_te
    R["dgm_tr_late"] = dgm_tr[-1]; R["dgm_te_late"] = dgm_te[-1]; R["ytr"] = ytr
    J = dict(run_id=rid, N=ntr, T=T, final={k: float(R[k][-1]) for k in ("spread_lab", "spread_km", "var_ratio", "effrank_ratio", "var_ratio_white", "nn_ratio", "dgm_ratio")},
             pathlen_ratio=float(np.median(pl_tr) / np.median(pl_te)), net_ratio=float(np.median(net_tr) / np.median(net_te)))
    # ---- per-sample memorization (AE only)
    if rec_tr is not None and rec_te is not None:
        R["err_tr"] = rec_tr.astype(np.float32); R["err_te"] = rec_te.astype(np.float32)
        mte = rec_te.mean(1); thr = MEMFRAC * mte.min(); imin = int(mte.argmin())
        below = rec_tr < thr; mem = below.any(0); tm = np.full(ntr, np.nan)
        for i in np.where(mem)[0]: tm[i] = steps[np.where(below[:, i])[0][0]]
        Npool = len(np.load(f"{rd}/train_index.npz")["indices"]) if os.path.isfile(f"{rd}/train_index.npz") else len(Xtr_all)
        ep = tm * 128 / Npool
        R["t_mem"] = tm; J.update(thr=float(thr), frac_mem_final=float(mem.mean()), imin_step=int(steps[imin]), frac_mem_at=[float(below[t].mean()) for t in (imin, T - 1)],
                                   frac_test_below_thr_final=float((rec_te[-1] < thr).mean()))
        if mem.sum() >= 10:
            J["t_mem_epoch_q"] = [float(np.nanpercentile(ep, q)) for q in (10, 50, 90)]
            ink = (Xtr ** 2).sum(1); Dp = A.sqd(Xtr); np.fill_diagonal(Dp, np.inf); dens = np.sqrt(np.sort(Dp, 1)[:, :5]).mean(1)
            ok = mem; early = rec_tr[imin]; ie = int(np.searchsorted(steps, 10 * Npool / 128)); ie = min(ie, T - 1)
            J["spearman_tmem"] = dict(ink=spearman(tm[ok], ink[ok]), pix_nn_dist=spearman(tm[ok], dens[ok]), err_at_testmin=spearman(tm[ok], early[ok]),
                                      dgm_epoch10=spearman(tm[ok], dgm_tr[ie][ok]), dgm_late=spearman(tm[ok], dgm_tr[-1][ok]), pathlen=spearman(tm[ok], pl_tr[ok]))
            J["tmem_by_class_epoch_median"] = [float(np.nanmedian(ep[(ytr == c) & ok])) if ((ytr == c) & ok).sum() > 2 else None for c in range(10)]
            # test-analogue: does test-point error at the min step predict its final error?
            J["spearman_te_err_min_vs_final"] = spearman(rec_te[imin], rec_te[-1]); J["spearman_tr_err_min_vs_final"] = spearman(rec_tr[imin], rec_tr[-1])
            # memorized vs not: latent path length / dgm
            J["early_vs_late_mem_pathlen"] = [float(np.median(pl_tr[tm <= np.nanmedian(tm)])), float(np.median(pl_tr[(tm > np.nanmedian(tm)) | np.isnan(tm)]))]
            J["spearman_err_final_vs_dgm_final"] = spearman(np.log(rec_tr[-1] + 1e-12), dgm_tr[-1])
            J["spearman_te_err_final_vs_dgm_final"] = spearman(rec_te[-1], dgm_te[-1])
    np.savez_compressed(f"{out}/{rid}.npz", **R); return J


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="mnist_ae_*,mnist_probe_none_d16_s[012]_phase_n250_wd0,mnist_probe_none_d16_s[012]_phase_n250_wd1e-4,mnist_probe_none_d16_s[012]_phase_n1000_wd0,mnist_probe_none_d16_s[012]_phase_n1000_wd1e-4,mnist_probe_none_d16_s[012]_phase_n4000_wd0,mnist_probe_none_d16_s[012]_phase_n4000_wd1e-4,mnist_probe_none_d16_s[012]_phase_nfull_wd0,mnist_probe_none_d16_s[012]_phase_nfull_wd1e-4"); ap.add_argument("--out", default=OUT); ap.add_argument("--cache", default=None); a = ap.parse_args()
    global CACHE
    if a.cache: CACHE = a.cache
    os.makedirs(a.out, exist_ok=True); dirs = A.resolve(a.runs); res = {}
    for d in dirs:
        rid = os.path.basename(d)
        if "tiny" in rid or not os.path.isdir(f"{d}/latents_train") or not os.path.isfile(f"{d}/model_final.pt"): continue
        if re.search(r"_n16000", rid): continue
        try:
            res[rid] = analyze(d, a.out); json.dump(res[rid], open(f"{a.out}/{rid}.json", "w")); print("done", rid, flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAIL", rid, repr(e), flush=True)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)


if __name__ == "__main__": main()
