"""Linear + nonlinear decomposition of the probe-latent motion:  z_{t+1,i} = A_t z_{t,i} + b_t + f_t(z_{t,i}, y_i) + eps.   torch (GPU optional).

Usage: python vfield_decomp.py run --device cuda --runs 'GLOB,...' --probes test[,train] [--nsplit 5] [--shard i/n] [--out DIR]   |   python vfield_decomp_aggregate.py
Fit on the TRAIN half, score on the held-out half (same 5 stratified splits as vfield.py). Displacement d_i = z_{t+1,i} - z_{t,i} (raw frame; A_t absorbs gauge).
Variance partition (held-out, fraction of sum ||d||^2, uncentered so translation counts as affine):
  R1  global affine  | R2  + class-wise affine (ridge, LOO-GCV) | R3  + nonlinear f on the residual (sequential: kNN10 and RBF kernel ridge) | unexplained = 1 - R3
  Joint: kernel ridge with linear + RBF kernel fitted to d directly (linear weight w_lin=1e3, i.e. nearly unpenalised affine part; RBF part class-agnostic (J_z) or class-aware (J_zy); LOO-chosen length scale and lambda).
  Nulls: nonlinear fits on residuals whose sample identity is shuffled among the train samples.
Why kernel ridge for the joint fit (and not an MLP with a linear skip): exact convex per-interval solve, batched over intervals, no optimiser noise or time-conditioning confound
(the MLP in vfield.py was worse than kNN/M2), explicit penalty only on the nonlinear component through the kernel weights.
Linear generator (all samples, descriptive): A_t from z_{t+1} = A z_t + b; L_t = logm(A_t)/dstep (fallback (A-I)/dstep when logm is complex / ill-conditioned); L = (tr/d) I + S + W.
Nonlinear field (descriptive, all samples): f1 = d - global affine, f2 = f1 - class-wise affine; magnitude vs margins, concentration near boundaries, directions, local divergence, PCA dimension.
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import argparse, glob, json, time
import numpy as np
import torch
from scipy.linalg import logm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vfield as V

OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/decomp"
DT = torch.float64
CHUNK = 30


def global_affine_fit(Z0, Z1, tr):
    """forward map fit on train indices: Z1 ~ [Z0 1] beta ; returns beta (K,d+1,d)."""
    X, Y = Z0[:, tr], Z1[:, tr]; K, m, d = X.shape; Xa = torch.cat([X, torch.ones(K, m, 1, dtype=X.dtype, device=X.device)], 2)
    G = Xa.transpose(1, 2) @ Xa; G = G + 1e-9 * G.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(d + 1, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(G, Xa.transpose(1, 2) @ Y)


def apply_affine_disp(Z0, beta):
    """displacement predicted by the forward affine map: [Z0 1] beta - Z0."""
    Za = torch.cat([Z0, torch.ones(Z0.shape[0], Z0.shape[1], 1, dtype=Z0.dtype, device=Z0.device)], 2); return Za @ beta - Z0


def krr(Ztr, Zte, Ytr, ctr, cte, w_lin, kappa, mults=(0.7, 1.5), lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Kernel ridge, batched over the leading (interval) axis, chunked. K = w_lin (z.z' + 1) + exp(-|z-z'|^2/(2 l^2)) (1 + kappa [c==c']).
    length scale l = mult * median pairwise distance, mult and lambda chosen per interval by closed-form LOO on the train half. Returns test predictions."""
    K, m, d = Ztr.shape; nte = Zte.shape[1]; out = torch.zeros(K, nte, d, dtype=Ztr.dtype, device=Ztr.device); same_tt = (ctr[:, None] == ctr[None, :]).to(Ztr.dtype); same_te = (cte[:, None] == ctr[None, :]).to(Ztr.dtype)
    for a in range(0, K, CHUNK):
        sl = slice(a, min(a + CHUNK, K)); Xt, Xe, Y = Ztr[sl], Zte[sl], Ytr[sl]; Dtt = torch.cdist(Xt, Xt) ** 2; Dte = torch.cdist(Xe, Xt) ** 2
        med = Dtt.flatten(1).median(1).values.clamp_min(1e-12); best_loo = None; best_pred = None
        for mult in mults:
            l2 = (mult ** 2 * med)[:, None, None]; Kt = torch.exp(-Dtt / (2 * l2)) * (1 + kappa * same_tt); Ke = torch.exp(-Dte / (2 * l2)) * (1 + kappa * same_te)
            if w_lin > 0:
                Kt = Kt + w_lin * (Xt @ Xt.transpose(1, 2) + 1.0); Ke = Ke + w_lin * (Xe @ Xt.transpose(1, 2) + 1.0)
            w, Vv = torch.linalg.eigh(Kt); VtY = Vv.transpose(1, 2) @ Y; V2 = Vv ** 2; scale = float(Kt.diagonal(dim1=1, dim2=2).mean()) if w_lin == 0 else 1.0
            lo = None
            for lam in lams:
                lamv = lam * (scale if w_lin == 0 else 1.0); inv = 1.0 / (w + lamv).clamp_min(1e-300); Mdiag = (V2 * inv[:, None, :]).sum(2); alpha = Vv @ (inv[:, :, None] * VtY)
                loo = ((alpha / Mdiag[:, :, None]) ** 2).sum((1, 2)); pred = Ke @ alpha
                if lo is None: lo, pr = loo, pred
                else:
                    msk = loo < lo; lo = torch.where(msk, loo, lo); pr = torch.where(msk[:, None, None], pred, pr)
            if best_loo is None: best_loo, best_pred = lo, pr
            else:
                msk = lo < best_loo; best_loo = torch.where(msk, lo, best_loo); best_pred = torch.where(msk[:, None, None], pr, best_pred)
        out[sl] = best_pred
    return out


def r2_0(pred, Dte):
    return (1 - ((Dte - pred) ** 2).sum((1, 2)) / ((Dte ** 2).sum((1, 2)) + 1e-300))


def partition(Z, steps, ycond, classes, dev, nsplit, light=False):
    T, n, d = Z.shape; K = T - 1; Z0, Z1 = Z[:-1], Z[1:]; D = Z1 - Z0; names = ["R1_affine", "R2_classaffine", "R3_knn", "R3_krr", "J_z", "J_zy", "R3_knn_null", "R3_krr_null", "J_zy_null_class"]
    out = np.zeros((nsplit, len(names), K), np.float32); cidx = torch.tensor([int(np.where(classes == c)[0][0]) for c in ycond], device=dev)
    for sp in range(nsplit):
        tr, te = V.make_split(ycond, sp); trt, tet = torch.tensor(tr, device=dev), torch.tensor(te, device=dev); gen = torch.Generator(device="cpu").manual_seed(500 + sp)
        beta = global_affine_fit(Z0, Z1, trt); p1_all = apply_affine_disp(Z0, beta); r1 = D - p1_all; Dte = D[:, tet]
        mu = Z0[:, trt].mean(1, keepdim=True); sig = ((Z0[:, trt] - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30; Zs = (Z0 - mu) / sig
        p1 = p1_all[:, tet]; out[sp, 0] = r2_0(p1, Dte).cpu().numpy()
        _, p2c, _ = V.class_models(Zs, r1, r1, ycond, tr, te, classes)         # class-wise affine on the global-affine residual
        out[sp, 1] = r2_0(p1 + p2c, Dte).cpu().numpy()
        # residual after global+class affine (train: in-sample residual of the class fit)
        _, p2c_tr, _ = V.class_models(Zs, r1, r1, ycond, tr, tr, classes); r2_tr = r1[:, trt] - p2c_tr; base_te = p1 + p2c
        dist = torch.cdist(Zs[:, tet], Zs[:, trt]); idx = dist.topk(10, largest=False).indices; fk = V.knn_pred(r2_tr, idx, 10); out[sp, 2] = r2_0(base_te + fk, Dte).cpu().numpy()
        Ztr_s, Zte_s = Zs[:, trt], Zs[:, tet]; ctr, cte = cidx[trt], cidx[tet]
        fkr = krr(Ztr_s, Zte_s, r2_tr, ctr, cte, 0.0, 2.0, **(dict(mults=(1.0,), lams=(1e-2, 1e-1, 1.0)) if light else {})); out[sp, 3] = r2_0(base_te + fkr, Dte).cpu().numpy()
        perm = torch.randperm(len(tr), generator=gen).to(dev); r2_sh = r2_tr[:, perm]
        out[sp, 6] = r2_0(base_te + V.knn_pred(r2_sh, idx, 10), Dte).cpu().numpy()
        if light: continue    # light mode: skip shuffled-sample kernel null and the joint kernel fits (columns stay 0)
        out[sp, 7] = r2_0(base_te + krr(Ztr_s, Zte_s, r2_sh, ctr, cte, 0.0, 2.0), Dte).cpu().numpy()
        Dtr = D[:, trt]; out[sp, 4] = r2_0(krr(Ztr_s, Zte_s, Dtr, ctr, cte, 1e3, 0.0), Dte).cpu().numpy(); out[sp, 5] = r2_0(krr(Ztr_s, Zte_s, Dtr, ctr, cte, 1e3, 2.0), Dte).cpu().numpy()
        cperm = cte[torch.randperm(len(te), generator=gen).to(dev)]; out[sp, 8] = r2_0(krr(Ztr_s, Zte_s, Dtr, ctr, cperm, 1e3, 2.0), Dte).cpu().numpy()   # class labels of test points shuffled
    return names, out


def generator(Z, steps, classes_unused=None):
    """A_t (forward affine, all samples), L = logm(A)/dstep, decomposition; numpy."""
    T, n, d = Z.shape; K = T - 1; rows = np.full((K, 9), np.nan); fall = np.zeros(K, bool)
    for k in range(K):
        X = np.concatenate([Z[k], np.ones((n, 1))], 1); beta = np.linalg.lstsq(X, Z[k + 1], rcond=None)[0]; A = beta[:d].T; ds = float(steps[k + 1] - steps[k])
        L = None
        try:
            if np.linalg.cond(A) < 1e8:
                Lm = logm(A)
                if np.all(np.isfinite(Lm)) and np.abs(Lm.imag).max() < 1e-6 * (np.abs(Lm.real).max() + 1e-30): L = Lm.real / ds
        except Exception: L = None
        if L is None: L = (A - np.eye(d)) / ds; fall[k] = True
        iso = np.trace(L) / d; Sy = (L + L.T) / 2 - iso * np.eye(d); W = (L - L.T) / 2; ev = np.linalg.eigvalsh((L + L.T) / 2)
        sv = np.linalg.svd(A, compute_uv=False)
        rows[k] = [iso, np.linalg.norm(Sy) / np.sqrt(d), np.linalg.norm(W) / np.sqrt(d), ev.max(), ev.min(), np.log(sv[0] / sv[-1]), np.log(sv).mean() / ds, np.abs(np.linalg.eigvals(L).imag).max(), np.linalg.norm(beta[d]) / ds]
    return rows, fall


GEN_COLS = ["iso_expansion", "shear_norm", "rotation_norm", "sym_eig_max", "sym_eig_min", "log_cond_A", "mean_log_sv_per_step", "max_abs_imag_eig", "translation_norm"]


def ranks(x): return x.argsort(1).argsort(1).to(x.dtype)


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); ra = ra - ra.mean(1, keepdim=True); rb = rb - rb.mean(1, keepdim=True); return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-300)


def nonlinear_field(Z, steps, ycond, ytrue, classes, rd, dev, bad_mask):
    T, n, d = Z.shape; K = T - 1; Z0, Z1 = Z[:-1], Z[1:]; D = Z1 - Z0; allidx = torch.arange(n, device=dev); beta = global_affine_fit(Z0, Z1, allidx); f1 = D - apply_affine_disp(Z0, beta)
    mu = Z0.mean(1, keepdim=True); sig = ((Z0 - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30; Zs = (Z0 - mu) / sig; res = {}
    clsidx = torch.tensor([int(np.where(classes == c)[0][0]) for c in ycond], device=dev)
    # class-wise affine (all samples) on f1 -> f2
    f2 = f1.clone()
    for ci, c in enumerate(classes):
        ii = torch.tensor(np.where(ycond == c)[0], device=dev)
        if len(ii) < 4: continue
        xb, db, A = V.ridge_fit(Zs[:, ii], f1[:, ii]); f2[:, ii] = f1[:, ii] - (db[:, None] + (Zs[:, ii] - xb[:, None]) @ A)
    # class means (clean points when labels are noisy) and geometry
    clean = ~bad_mask; cm = torch.stack([Z0[:, torch.tensor(np.where(clean & (ycond == c))[0], device=dev)].mean(1) for c in classes], 1)   # (K,C,d)
    dist = torch.cdist(Z0, cm)                                                          # (K,n,C)
    own = dist.gather(2, clsidx[None, :, None].expand(K, n, 1))[:, :, 0]; dm = dist.clone(); dm.scatter_(2, clsidx[None, :, None].expand(K, n, 1), float("inf")); dother, jother = dm.min(2); margin = (dother - own) / sig[:, :, 0]
    fn1, fn2 = f1.norm(dim=2), f2.norm(dim=2)
    for nm, f, fn in (("f1", f1, fn1), ("f2", f2, fn2)):
        res[f"{nm}_norm_rms"] = (fn ** 2).mean(1).sqrt().cpu().numpy(); res[f"{nm}_over_disp"] = ((f ** 2).sum((1, 2)) / ((D ** 2).sum((1, 2)) + 1e-300)).cpu().numpy()
        res[f"{nm}_spearman_margin"] = spearman(fn, margin).cpu().numpy(); res[f"{nm}_spearman_dother"] = spearman(fn, dother / sig[:, :, 0]).cpu().numpy()
        lowm = margin <= margin.quantile(0.2, dim=1, keepdim=True); res[f"{nm}_energy_share_lowest20pct_margin"] = ((fn ** 2) * lowm).sum(1).div((fn ** 2).sum(1) + 1e-300).cpu().numpy()
        vown = cm[:, :, :][torch.arange(K, device=dev)[:, None], clsidx[None, :]] - Z0; voth = cm[torch.arange(K, device=dev)[:, None], jother] - Z0
        cos = lambda a, b: ((a * b).sum(2) / (a.norm(dim=2) * b.norm(dim=2) + 1e-300))
        res[f"{nm}_cos_own"] = cos(f, vown).mean(1).cpu().numpy(); res[f"{nm}_cos_nearest_other"] = cos(f, voth).mean(1).cpu().numpy()
        fc = f - f.mean(1, keepdim=True); sv = torch.linalg.svdvals(fc); ev = sv ** 2; evn = ev / ev.sum(1, keepdim=True); cs = evn.cumsum(1)
        res[f"{nm}_n90"] = (cs < 0.9).sum(1).add(1).cpu().numpy(); res[f"{nm}_participation_ratio"] = (ev.sum(1) ** 2 / (ev ** 2).sum(1)).cpu().numpy(); res[f"{nm}_top1_share"] = evn[:, 0].cpu().numpy()
        if bad_mask.any():
            bm = torch.tensor(bad_mask, device=dev); res[f"{nm}_norm_ratio_bad_over_clean"] = (fn[:, bm].mean(1) / (fn[:, ~bm].mean(1) + 1e-300)).cpu().numpy()
    res["margin_median"] = margin.median(1).values.cpu().numpy()
    # local divergence of f1 and f2 (kNN local linear, k=30, in raw units)
    kk = 30; Dz = torch.cdist(Z0, Z0); nn = Dz.topk(kk + 1, largest=False).indices[:, :, 1:]
    for nm, f in (("f1", f1), ("f2", f2)):
        div = torch.zeros(K, n, dtype=DT, device=dev)
        for a in range(0, K, 10):
            sl = slice(a, min(a + 10, K)); Zc, fc_, idx = Z0[sl], f[sl], nn[sl]; Kc = Zc.shape[0]; ar = torch.arange(Kc, device=dev)[:, None, None]
            dZ = Zc[ar, idx] - Zc[:, :, None, :]; dF = fc_[ar, idx] - fc_[:, :, None, :]                                        # (Kc,n,kk,d)
            G = dZ.transpose(2, 3) @ dZ; G = G + 1e-3 * G.diagonal(dim1=2, dim2=3).mean(2)[:, :, None, None] * torch.eye(d, dtype=DT, device=dev)
            J = torch.linalg.solve(G, dZ.transpose(2, 3) @ dF); div[sl] = J.diagonal(dim1=2, dim2=3).sum(2)
        res[f"{nm}_div_median"] = div.median(1).values.cpu().numpy(); res[f"{nm}_div_frac_pos"] = (div > 0).double().mean(1).cpu().numpy()
    # final-head quantities (only meaningful in the last intervals; skipped when the latent dimension differs from the head, e.g. projected CIFAR latents)
    try:
        sd = torch.load(f"{rd}/model_final.pt", map_location="cpu", weights_only=False); W = sd["probe.weight"].double().to(dev); b = sd["probe.bias"].double().to(dev); L = 15
        logits = Z0[-L:] @ W.T + b; yl = torch.tensor(ycond, device=dev); lp = torch.log_softmax(logits, -1); loss = -lp.gather(2, yl[None, :, None].expand(L, n, 1))[:, :, 0]; conf = lp.max(2).values.exp()
        lt = logits.clone(); lt.scatter_(2, yl[None, :, None].expand(L, n, 1), float("-inf")); hmargin = logits.gather(2, yl[None, :, None].expand(L, n, 1))[:, :, 0] - lt.max(2).values
        for nm, fn in (("f1", fn1), ("f2", fn2)):
            for qn, q in (("loss", loss), ("conf", conf), ("hmargin", hmargin)):
                r = np.full(K, np.nan); r[-L:] = spearman(fn[-L:], q).cpu().numpy(); res[f"{nm}_spearman_head_{qn}"] = r

    except Exception:
        for nm in ("f1", "f2"):
            for qn in ("loss", "conf", "hmargin"): res[f"{nm}_spearman_head_{qn}"] = np.full(K, np.nan)
    # mislabeled points: direction of f only (global affine removed; f2 also class affine removed)
    if bad_mask.any():
        bm = torch.tensor(np.where(bad_mask)[0], device=dev); ytr = torch.tensor([int(np.where(classes == c)[0][0]) for c in ytrue], device=dev)
        for nm, f in (("f1", f1), ("f2", f2)):
            vg = cm[torch.arange(K, device=dev)[:, None], clsidx[bm][None, :]] - Z0[:, bm]; vt = cm[torch.arange(K, device=dev)[:, None], ytr[bm][None, :]] - Z0[:, bm]; fb = f[:, bm]
            cos = lambda a, b: ((a * b).sum(2) / (a.norm(dim=2) * b.norm(dim=2) + 1e-300))
            res[f"mis_{nm}_cos_given"] = cos(fb, vg).mean(1).cpu().numpy(); res[f"mis_{nm}_cos_true"] = cos(fb, vt).mean(1).cpu().numpy()
        res["n_bad"] = np.array(int(bad_mask.sum()))
    return res


def process(rd, probe, dev, nsplit, tmax=0):
    Z, steps, ytrue, noisy = V.load_run(rd, probe)
    if tmax: Z, steps = Z[:tmax], steps[:tmax]   # dev only
    ycond = noisy if noisy is not None else ytrue; classes = np.unique(ycond); bad = (ytrue != ycond) if noisy is not None else np.zeros(len(ytrue), bool)
    Zt = torch.tensor(Z, dtype=DT, device=dev); t0 = time.time(); names, P = partition(Zt, steps, ycond, classes, dev, nsplit)
    G, fall = generator(Z, steps); nl = nonlinear_field(Zt, steps, ycond, ytrue, classes, rd, dev, bad)
    out = dict(steps=steps, part_names=np.array(names), part_mean=P.mean(0), part_std=P.std(0), gen=G, gen_cols=np.array(GEN_COLS), gen_fallback=fall, n=np.array(Z.shape[1]), min_class=np.array(min(int((ycond == c).sum()) for c in classes)))
    for k, v in nl.items(): out[f"nl_{k}"] = v
    out["seconds"] = np.array(time.time() - t0); return out


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); r = sp.add_parser("run")
    r.add_argument("--device", default="cpu"); r.add_argument("--runs", default="mnist_probe_none_d16_s*_phase_*"); r.add_argument("--probes", default="test"); r.add_argument("--out", default=OUT)
    r.add_argument("--nsplit", type=int, default=5); r.add_argument("--limit", type=int, default=0); r.add_argument("--shard", default=""); r.add_argument("--force", action="store_true"); r.add_argument("--tmax", type=int, default=0)
    a = ap.parse_args(); dev = torch.device(a.device); torch.set_num_threads(8)
    assert dev.type == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "0"), "pin CUDA_VISIBLE_DEVICES to an idle GPU (never 0)"
    os.makedirs(f"{a.out}/runs", exist_ok=True); os.makedirs(f"{a.out}/claims", exist_ok=True)
    dirs = sorted(d for g in a.runs.split(",") for d in glob.glob(f"{V.RUNS}/{g}") if os.path.isdir(d) and os.path.isfile(f"{d}/model_final.pt"))
    if a.shard: i, n = map(int, a.shard.split("/")); dirs = dirs[i::n]
    if a.limit: dirs = dirs[:a.limit]
    for d in dirs:
        rid = os.path.basename(d)
        for probe in a.probes.split(","):
            fo = f"{a.out}/runs/{rid}_{probe}.npz"
            if os.path.isfile(fo) and not a.force: continue
            if probe == "train" and (not os.path.isdir(f"{d}/latents_train") or "_gap_" not in rid): continue
            try: os.makedirs(f"{a.out}/claims/{rid}_{probe}")
            except FileExistsError: continue
            res = process(d, probe, dev, a.nsplit, a.tmax); np.savez_compressed(fo, **res); print(f"{rid} {probe} done {float(res['seconds']):.0f}s", flush=True)
            if dev.type == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__": main()
