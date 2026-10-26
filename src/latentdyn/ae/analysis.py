"""Autoencoder (AE) latent-dynamics analysis, per run.   CPU only (torch + numpy, threads capped at 8).

Usage:
  python -m latentdyn.ae.analysis run --runs 'GLOB[,GLOB...]' [--out DIR] [--nsplit 3] [--force] [--no_vfield] [--limit N] [--shard i/n]
  (GLOB searched under ae_dynamics/runs, runs, runs_v2/runs, artifacts root, or absolute; runs without model_final.pt are skipped)
Outputs (per run):  OUT/runs/<run_id>.npz (time series) + OUT/runs/<run_id>.json (scalar summary).  OUT default = /vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis

Labels are ANNOTATIONS only for an AE (the encoder never sees them).  Every class-based metric ("lab") is paired with a LABEL-FREE twin ("km") using a fixed
input-space grouping: k-means (k=10) on 50-d pixel-PCA, fitted once per dataset on the full test split (cached in OUT/_cache) and applied to the probe images.
GAUGE: AE latents are defined only up to invertible maps absorbed by the decoder (nonlinear AE: any invertible re-parametrisation; linear AE: GL(d)).  Hence we use only
quantities invariant (or nearly) to affine/linear re-coordinatisation: affine-removed residual fractions (R2), ratios of traces (Fisher, spread ratios), principal angles,
linear CKA, affine readout R2, kNN-graph overlaps, effective ranks (invariant to rotations, not to anisotropic rescaling -- flagged '*').

Time series saved (T snapshots; prefix te_ = test probe, tr_ = train probe; lab_ / km_ = label groups / input-cluster groups):
  {set}_{g}_fisher (tr Sw / tr Sb, never pinv-NC1), {set}_{g}_ncm (nearest-class-mean acc w.r.t. the group), {set}_{g}_knn (LOO kNN k=10 group agreement), {set}_effrank* ,
  {set}_latkm_ari / _nmi (k-means(10) ON THE LATENTS vs labels: label structure emerging), {set}_knn_overlap (frac of 10 input-space NN preserved in latent), {set}_cka_pca (linear CKA(Z, top-d PC scores)),
  {set}_pa_mean_cos2 / _pa_min_cos (principal angles latent span vs top-d PCA subspace of the TRAINING POOL), {set}_pc_r2 (T,K) per-PC affine-readout R2 (cross-fit),
  {set}_aff_recon_frac (variance fraction of pixels explained by affine readout of Z; compare pca_bound), recon_* (logged scalars), {set}_recon_{med,mean,p90,tail10} (per-sample logged recon_err),
  gen_* (train/test geometry): spread_ratio_{lab,km} (within-group spread train/test), var_ratio, effrank_ratio, te2tr_nn (median test->nearest train latent / median train-train NN),
  vf_{set}_{g}_{R1,R2,R3_knn,R3_krr,R3_knn_null} (vfield_decomp.partition, light), vf_transfer_{g} (vfield_traintest.transfer_block mean over splits, shape (gm,src,tgt,(g,c),K)).
"""
import os
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, json, time, hashlib
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

ROOT = "/vol/bitbucket/lbastian/latent-dynamics"
AE_RUNS = f"{ROOT}/ae_dynamics/runs"
OUT = f"{ROOT}/ae_dynamics/analysis"
NG = 10   # k-means groups
KPC = 32  # PCs tracked for staging


# ------------------------------------------------------------------ data / groupings
def load_images(dataset, train):
    """normalised flat float32 images (N, D), identical to data._transform + flatten."""
    from torchvision import datasets
    from latentdyn import data as D
    cls = D._DATASET_CLASSES[dataset]; ds = cls(root=f"{ROOT}/data", train=train, download=False)
    x = np.asarray(ds.data).astype(np.float32) / 255.0
    if x.ndim == 3: x = x[:, None]
    else: x = x.transpose(0, 3, 1, 2)
    mean, std = D._NORMALIZATION[dataset]; mean = np.array(mean, np.float32)[None, :, None, None]; std = np.array(std, np.float32)[None, :, None, None]
    return ((x - mean) / std).reshape(len(x), -1), np.asarray(ds.targets)


_IMG = {}
def images(dataset, train):
    k = (dataset, train)
    if k not in _IMG: _IMG[k] = load_images(dataset, train)
    return _IMG[k]


def group_model(dataset, cache):
    """fixed per-dataset input grouping: 50-d pixel PCA + k-means(10), fitted on the full test split. returns assign(X)->labels."""
    f = f"{cache}/{dataset}_groups.npz"
    if not os.path.isfile(f):
        from sklearn.cluster import KMeans
        X, _ = images(dataset, False); mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); P = Vt[:50]
        km = KMeans(NG, n_init=10, random_state=0).fit((X - mu) @ P.T)
        os.makedirs(cache, exist_ok=True); np.savez(f, mu=mu, P=P, C=km.cluster_centers_)
    z = np.load(f)
    def assign(X):
        s = (X - z["mu"]) @ z["P"].T; return ((s[:, None, :] - z["C"][None]) ** 2).sum(-1).argmin(1)
    return assign


def pca_pool(dataset, pool_idx, cache):
    """PCA (centered) of the training pool: mean, top-KPC eigvecs (D,KPC), eigenvalues (top KPC) and total variance."""
    key = hashlib.md5(np.asarray(pool_idx).tobytes()).hexdigest()[:10]; f = f"{cache}/{dataset}_pca_{key}.npz"
    if not os.path.isfile(f):
        X, _ = images(dataset, True); X = X[pool_idx].astype(np.float64); mu = X.mean(0); Xc = X - mu
        C = Xc.T @ Xc / len(X); w, V = np.linalg.eigh(C); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
        os.makedirs(cache, exist_ok=True); np.savez(f, mu=mu, V=V[:, :KPC], w=w[:KPC], tot=w.sum())
    z = np.load(f); return z["mu"], z["V"], z["w"], float(z["tot"])


# ------------------------------------------------------------------ small helpers
def load_snaps(rd, sub):
    fs = sorted(glob.glob(f"{rd}/{sub}/step_*.npz")); steps, Z, rec, sc = [], [], [], {}
    y = None
    for f in fs:
        with np.load(f) as z:
            steps.append(int(z["step"])); Z.append(z["latents"].astype(np.float64)); y = z["labels"].astype(np.int64) if y is None else y
            rec.append(z["recon_err"].astype(np.float64) if "recon_err" in z.files else None)
            for k in z.files:
                if k in ("latents", "labels", "recon_err", "noisy_labels") or z[k].ndim != 0: continue
                sc.setdefault(k, []).append(float(z[k]))
    o = np.argsort(steps); steps = np.array(steps)[o]; Z = np.stack(Z)[o]
    recs = None if any(r is None for r in rec) else np.stack([rec[i] for i in o])
    return steps, Z, y, recs, {k: np.array(v)[o] for k, v in sc.items() if len(v) == len(o)}


def effrank(Z):
    """participation ratio of centered singular values^2 per snapshot, (T,n,d)->(T,)"""
    s = np.linalg.svd(Z - Z.mean(1, keepdims=True), compute_uv=False) ** 2; return s.sum(1) ** 2 / (s ** 2).sum(1)


def sw_sb(Z, g, groups):
    """(T,) tr(Sw), tr(Sb) using population-weighted groups."""
    T, n, d = Z.shape; gm = Z.mean(1); sw = np.zeros(T); sb = np.zeros(T)
    for c in groups:
        ii = np.where(g == c)[0]
        if len(ii) == 0: continue
        zc = Z[:, ii]; mc = zc.mean(1); p = len(ii) / n; sw += p * ((zc - mc[:, None]) ** 2).sum(2).mean(1); sb += p * ((mc - gm) ** 2).sum(1)
    return sw, sb


def ncm(Z, g, groups):
    mus = np.stack([Z[:, g == c].mean(1) for c in groups], 1); d = ((Z[:, :, None, :] - mus[:, None]) ** 2).sum(-1)
    return (np.asarray(groups)[d.argmin(2)] == g[None]).mean(1)


def sqd(X, Y=None):
    Y = X if Y is None else Y; return np.maximum((X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * X @ Y.T, 0)


def knn_idx(X, k):
    D = sqd(X); np.fill_diagonal(D, np.inf); return np.argpartition(D, k, axis=1)[:, :k]


def knn_agree(Z, g, k=10):
    out = np.zeros(len(Z))
    for t in range(len(Z)):
        idx = knn_idx(Z[t], k); out[t] = (g[idx] == g[:, None]).mean()   # mean fraction of neighbours in the same group (-> LOO kNN agreement)
    return out


def knn_overlap(Z, nn_in, k=10):
    out = np.zeros(len(Z))
    for t in range(len(Z)):
        idx = knn_idx(Z[t], k); out[t] = np.mean([len(set(idx[i]) & set(nn_in[i])) / k for i in range(len(idx))])
    return out


def latent_kmeans_nmi(Z, y, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari, nmi = np.zeros(len(Z)), np.zeros(len(Z))
    for t in range(len(Z)):
        lab = KMeans(NG, n_init=2, random_state=seed).fit_predict(Z[t]); ari[t] = adjusted_rand_score(y, lab); nmi[t] = normalized_mutual_info_score(y, lab)
    return ari, nmi


def orth(A):
    U, S, _ = np.linalg.svd(A, full_matrices=False); return U[:, S > 1e-10 * S.max()] if S.max() > 0 else U[:, :0]


def principal_angles(Z, S):
    """Z (T,n,d) latents, S (n,d') PC scores; cos of principal angles between span(centered Z) and span(centered S) per snapshot -> (T, min)."""
    Sc = orth(S - S.mean(0)); out = np.zeros((len(Z), min(Z.shape[2], Sc.shape[1])))
    for t in range(len(Z)):
        U = orth(Z[t] - Z[t].mean(0)); c = np.linalg.svd(U.T @ Sc, compute_uv=False)
        out[t, :len(c)] = np.clip(c[:out.shape[1]], 0, 1)
    return out


def cka_lin(A, B):
    A = A - A.mean(0); B = B - B.mean(0); return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B) + 1e-300))


def affine_readout_r2(Z, Y, nf=2, seed=0):
    """cross-fit (nf folds) affine regression Y ~ [Z 1]; per-column R2 (T, K)."""
    T, n, d = Z.shape; perm = np.random.RandomState(seed).permutation(n); folds = np.array_split(perm, nf); K = Y.shape[1]; sse = np.zeros((T, K)); Yc = Y - Y.mean(0); sst = (Yc ** 2).sum(0)
    Za = np.concatenate([Z, np.ones((T, n, 1))], 2)
    for f in range(nf):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(nf) if j != f]); X = Za[:, tr]; G = X.transpose(0, 2, 1) @ X
        G = G + 1e-8 * np.trace(G, axis1=1, axis2=2)[:, None, None] / (d + 1) * np.eye(d + 1); B = np.linalg.solve(G, X.transpose(0, 2, 1) @ Y[tr]); sse += ((Za[:, te] @ B - Y[te]) ** 2).sum(1)
    return 1 - sse / sst


def rise_steps(steps, r, frac=0.5, mingain=0.05):
    """per column: step (interp in log1p time) where the series first reaches r0 + frac*(final-r0) (r0 = value at step 0, final = mean of last 3 snaps); NaN if final-r0 < mingain.
    Baseline-subtracted because a random-init encoder already reads out the top PCs to some extent."""
    T, K = r.shape; out = np.full(K, np.nan); x = np.log1p(steps.astype(float))
    for k in range(K):
        r0 = r[0, k]; fin = r[-3:, k].mean()
        if fin - r0 < mingain: continue
        thr = r0 + frac * (fin - r0); i = np.where(r[:, k] >= thr)[0]
        if len(i) == 0: continue
        i = i[0]
        out[k] = steps[0] if i == 0 else float(np.expm1(x[i - 1] + (thr - r[i - 1, k]) / (r[i, k] - r[i - 1, k] + 1e-300) * (x[i] - x[i - 1])))
    return out


from latentdyn.stats import spearman  # noqa: F401  (re-exported as A.spearman)


def final_recon(rd, cfg, sets, cache=None):
    """final-model recon MSE (normalised pixels) on full test split vs train pool subset (<=10000 each) + per-sample on probes. returns dict or {}."""
    try:
        from latentdyn import model as MD
        from latentdyn.data import _NORMALIZATION
        ds = cfg["dataset"]; inc = 3 if ds == "cifar10" else 1; img = 32 if ds == "cifar10" else 28
        m = MD.LatentProbeModel(cfg["latent_dim"], encoder_arch=cfg.get("encoder_arch", "mlp"), objective=cfg["objective"], latent_norm=cfg.get("latent_norm", "none"),
                                in_channels=cfg.get("in_channels", inc), img_size=cfg.get("img_size", img), linear_ae=bool(cfg.get("linear_ae", False)))
        m.load_state_dict(torch.load(f"{rd}/model_final.pt", map_location="cpu", weights_only=False)); m.eval(); out = {}
        def rec(X):
            r = []
            with torch.no_grad():
                for i in range(0, len(X), 2000):
                    x = torch.tensor(X[i:i + 2000]).reshape(-1, cfg.get("in_channels", inc), cfg.get("img_size", img), cfg.get("img_size", img)); _, rc = m(x); r.append(((rc - x.flatten(1)) ** 2).mean(1).numpy())
            return np.concatenate(r)
        Xte, _ = images(ds, False); Xtr, _ = images(ds, True); rs = np.random.RandomState(0)
        pool = sets["pool"]; sub = rs.choice(pool, min(10000, len(pool)), replace=False); ete = rec(Xte[rs.choice(len(Xte), min(10000, len(Xte)), replace=False)]); etr = rec(Xtr[sub])
        out = dict(final_recon_train=float(etr.mean()), final_recon_test=float(ete.mean()), final_recon_gap=float(ete.mean() / etr.mean() - 1))
        if cfg.get("linear_ae"):   # linear AE: decoder / encoder subspaces vs top-d PCA of the training pool
            mu, V, w, tot = pca_pool(ds, pool, cache or f"{OUT}/_cache"); d = cfg["latent_dim"]; Wd = m.decoder.net.weight.detach().double().numpy(); We = m.encoder.net[1].weight.detach().double().numpy()
            for nm, A in (("dec", Wd), ("enc", We.T)):
                c = np.linalg.svd(orth(A).T @ V[:, :d], compute_uv=False); out[f"final_lin_{nm}_pca_meancos2"] = float((c ** 2).mean()); out[f"final_lin_{nm}_pca_mincos"] = float(c.min())
        return out
    except Exception as e:
        return {"final_recon_error": repr(e)[:200]}


# ------------------------------------------------------------------ main per-run analysis
def analyze(rd, nsplit=3, vfield=True, out_dir=OUT):
    t0 = time.time(); cfg = json.load(open(f"{rd}/config.json")); ds = cfg["dataset"]; cache = f"{out_dir}/_cache"; R = {}; J = {}
    steps, Zte, yte, rec_te, sc = load_snaps(rd, "latents"); has_tr = os.path.isdir(f"{rd}/latents_train") and len(glob.glob(f"{rd}/latents_train/step_*.npz")) == len(steps)
    if has_tr: _, Ztr, ytr, rec_tr, _ = load_snaps(rd, "latents_train")
    T, _, d = Zte.shape; R["steps"] = steps
    for k, v in sc.items():
        if k.startswith("recon") or k in ("test_loss", "train_loss", "test_acc", "train_acc"): R[f"sc_{k}"] = v
    # probe images / pools
    Xall_te, _ = images(ds, False); Xall_tr, _ = images(ds, True); pi = np.load(f"{rd}/probe_index.npz")["indices"]; Xte = Xall_te[pi]
    pool = np.load(f"{rd}/train_index.npz")["indices"] if os.path.isfile(f"{rd}/train_index.npz") else np.arange(len(Xall_tr))
    sets = {"te": dict(Z=Zte, y=yte, X=Xte, rec=rec_te)}
    if has_tr: ti = np.load(f"{rd}/probe_train_index.npz")["indices"]; sets["tr"] = dict(Z=Ztr, y=ytr, X=Xall_tr[ti], rec=rec_tr)
    assign = group_model(ds, cache); mu, V, w, tot = pca_pool(ds, pool, cache); dd = min(d, V.shape[1])
    classes = np.arange(10); kmg = np.arange(NG)
    J.update(dict(run_id=cfg["run_id"], dataset=ds, objective=cfg["objective"], latent_dim=d, linear_ae=bool(cfg.get("linear_ae", False)), N=int(cfg.get("train_subset") or len(pool)), weight_decay=cfg.get("weight_decay", 0.0), seed=cfg["seed"], T=int(T), has_train_probe=bool(has_tr), pool=int(len(pool)),
                  pca_bound=float(w[:dd].sum() / tot)))
    for s, S in sets.items():
        Z, X = S["Z"], S["X"]; S["g"] = {"lab": S["y"], "km": assign(X)}; S["gs"] = {"lab": classes, "km": kmg}
        # static input-space references
        Pc = (X - mu) @ V[:, :50 if V.shape[1] >= 50 else V.shape[1]]
        for gn in ("lab", "km"):
            g, gs = S["g"][gn], S["gs"][gn]; sw, sb = sw_sb(Z, g, gs); R[f"{s}_{gn}_fisher"] = sw / sb; R[f"{s}_{gn}_ncm"] = ncm(Z, g, gs); R[f"{s}_{gn}_knn"] = knn_agree(Z, g)
            sw0, sb0 = sw_sb(Pc[None], g, gs); J[f"{s}_{gn}_fisher_pixelpca"] = float(sw0[0] / sb0[0]); J[f"{s}_{gn}_knn_pixel"] = float(knn_agree(X[None].astype(np.float64), g)[0])
        R[f"{s}_effrank"] = effrank(Z)
        Sc = (X - mu) @ V[:, :dd]; nn_in = knn_idx(X.astype(np.float64), 10)
        R[f"{s}_knn_overlap"] = knn_overlap(Z, nn_in); R[f"{s}_cka_pca"] = np.array([cka_lin(Z[t], Sc) for t in range(T)])
        pa = principal_angles(Z, Sc); R[f"{s}_pa_mean_cos2"] = (pa ** 2).mean(1); R[f"{s}_pa_min_cos"] = pa.min(1)
        if s == "te" or True:
            ari, nmi = latent_kmeans_nmi(Z, S["y"]); R[f"{s}_latkm_ari"] = ari; R[f"{s}_latkm_nmi"] = nmi
        Yk = (X - mu) @ V; r = affine_readout_r2(Z, Yk); R[f"{s}_pc_r2"] = r; R[f"{s}_aff_recon_frac"] = (r * (w / tot)[None]).sum(1)
        rs_ = rise_steps(steps, r); R[f"{s}_pc_rise"] = rs_; J[f"{s}_pc_rise_spearman_dd"] = spearman(np.arange(dd)[:], rs_[:dd]); J[f"{s}_pc_n_learned_dd"] = int(np.isfinite(rs_[:dd]).sum())
        if S["rec"] is not None:
            e = S["rec"]; R[f"{s}_recon_med"] = np.median(e, 1); R[f"{s}_recon_mean"] = e.mean(1); R[f"{s}_recon_p90"] = np.quantile(e, 0.9, axis=1)
            srt = np.sort(e, 1); k = max(1, e.shape[1] // 10); R[f"{s}_recon_tail10"] = srt[:, -k:].sum(1) / srt.sum(1)
            dens = np.sort(sqd(X.astype(np.float64)), 1)[:, 1:11].mean(1); R[f"{s}_recon_spearman_density"] = np.array([spearman(e[t], dens) for t in range(T)])
    J["pc_rise_final_note"] = "pc_rise: step at which per-PC affine-readout R2 first reaches r0+50% of (final-r0) (NaN if gain<0.05)"
    # logged recon gap
    if "sc_recon_train" in R and "sc_recon_test" in R: R["recon_gap_rel"] = R["sc_recon_test"] / R["sc_recon_train"] - 1
    # train vs test geometry
    if has_tr:
        for gn in ("lab", "km"):
            wte = sw_sb(sets["te"]["Z"], sets["te"]["g"][gn], sets["te"]["gs"][gn])[0]; wtr = sw_sb(sets["tr"]["Z"], sets["tr"]["g"][gn], sets["tr"]["gs"][gn])[0]; R[f"gen_spread_ratio_{gn}"] = wtr / wte
        vt = lambda Z: ((Z - Z.mean(1, keepdims=True)) ** 2).sum(2).mean(1)
        R["gen_var_ratio"] = vt(Ztr) / vt(Zte); R["gen_effrank_ratio"] = R["tr_effrank"] / R["te_effrank"]
        a = np.zeros(T)
        for t in range(T):
            Dx = sqd(Zte[t], Ztr[t]); dtt = sqd(Ztr[t]); np.fill_diagonal(dtt, np.inf)
            a[t] = np.sqrt(np.median(Dx.min(1))) / (np.sqrt(np.median(dtt.min(1))) + 1e-300)
        R["gen_te2tr_nn"] = a
        if "te_recon_mean" in R and "tr_recon_mean" in R: R["gen_probe_recon_ratio"] = R["te_recon_mean"] / R["tr_recon_mean"]
    # vector-field decomposition (reusing vfield_decomp.partition / vfield_traintest.transfer_block); groups plug in as the "class" argument
    if vfield:
        from latentdyn.vfield import decomp as VD, traintest as VT
        dev = torch.device("cpu"); Zt = {s: torch.tensor(S["Z"], dtype=torch.float64) for s, S in sets.items()}
        for s, S in sets.items():
            for gn in ("lab", "km"):
                g = S["g"][gn]
                names, P = VD.partition(Zt[s], steps, g, S["gs"][gn], dev, nsplit, light=True)
                for nm in ("R1_affine", "R2_classaffine", "R3_knn", "R3_krr", "R3_knn_null"):
                    R[f"vf_{s}_{gn}_{nm}"] = P[:, names.index(nm)].mean(0)
        if has_tr:
            for gn in ("lab", "km"):
                SS = {s: dict(Z0=Zt[s][:-1], Z1=Zt[s][1:], y=sets[s]["g"][gn]) for s in sets}
                R[f"vf_transfer_{gn}"] = VT.transfer_block(SS, nsplit, dev, sets["te"]["gs"][gn]).mean(0)
    # summaries
    def at(x, f=-1): return float(x[f])
    for k in ("te_lab_fisher", "te_km_fisher", "te_lab_ncm", "te_km_ncm", "te_lab_knn", "te_km_knn", "te_latkm_ari", "te_latkm_nmi", "te_knn_overlap", "te_cka_pca", "te_pa_mean_cos2", "te_pa_min_cos", "te_aff_recon_frac", "te_effrank",
              "tr_lab_fisher", "tr_lab_ncm", "tr_latkm_nmi", "tr_pa_mean_cos2", "tr_aff_recon_frac", "gen_spread_ratio_lab", "gen_spread_ratio_km", "gen_var_ratio", "gen_effrank_ratio", "gen_te2tr_nn", "recon_gap_rel", "gen_probe_recon_ratio",
              "sc_recon_train", "sc_recon_test", "te_recon_tail10", "tr_recon_tail10"):
        if k in R: J[f"final_{k}"] = at(R[k])
    for k in ("te_lab_fisher", "te_lab_ncm", "te_latkm_nmi"):
        if k in R: J[f"initial_{k}"] = at(R[k], 0)
    for gn in ("lab", "km"):
        for s in sets:
            kk = f"vf_{s}_{gn}_R2_classaffine"
            if kk in R: J[f"vf_{s}_{gn}_classaffine_gain_late"] = float((R[kk] - R[f"vf_{s}_{gn}_R1_affine"])[-max(3, T // 4):].mean())
            if kk in R: J[f"vf_{s}_{gn}_R3krr_late"] = float(R[f"vf_{s}_{gn}_R3_krr"][-max(3, T // 4):].mean())
    J.update(final_recon(rd, cfg, dict(pool=pool), cache)); J["seconds"] = time.time() - t0
    return R, J


def resolve(globs):
    dirs = []
    for g in globs.split(","):
        for base in (AE_RUNS, f"{ROOT}/runs", f"{ROOT}/runs_v2/runs", ROOT, ""):
            hit = glob.glob(os.path.join(base, g));
            if hit: dirs += hit; break
    return sorted(set(d for d in dirs if os.path.isdir(d) and os.path.isfile(f"{d}/config.json")))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); r = sp.add_parser("run")
    r.add_argument("--runs", required=True); r.add_argument("--out", default=OUT); r.add_argument("--nsplit", type=int, default=3); r.add_argument("--force", action="store_true")
    r.add_argument("--no_vfield", action="store_true"); r.add_argument("--limit", type=int, default=0); r.add_argument("--shard", default="")
    a = ap.parse_args(); os.makedirs(f"{a.out}/runs", exist_ok=True)
    dirs = resolve(a.runs); print(f"{len(dirs)} run dirs matched", flush=True)
    if a.shard: i, n = map(int, a.shard.split("/")); dirs = dirs[i::n]
    if a.limit: dirs = dirs[:a.limit]
    for d in dirs:
        rid = os.path.basename(d.rstrip("/")); cfgo = json.load(open(f"{d}/config.json"))
        if not os.path.isfile(f"{d}/model_final.pt"): print(f"skip {rid}: no model_final.pt", flush=True); continue
        if os.path.isfile(f"{a.out}/runs/{rid}.json") and not a.force: print(f"cached {rid}", flush=True); continue
        try:
            R, J = analyze(d, a.nsplit, not a.no_vfield, a.out)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"FAIL {rid}: {e!r}", flush=True); continue
        np.savez_compressed(f"{a.out}/runs/{rid}.npz", **R); json.dump(J, open(f"{a.out}/runs/{rid}.json", "w"), indent=1, default=float)
        print(f"{rid} done {J['seconds']:.0f}s", flush=True)


if __name__ == "__main__": main()
