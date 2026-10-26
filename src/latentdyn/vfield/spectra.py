"""Spectra of the class-wise affine generators of the probe-latent motion (step 1 of 3). torch fits (GPU optional), numpy eigen-analysis (CPU, one process).
Usage: python -m latentdyn.vfield.spectra run --device cuda --runs 'GLOB,...' --probes test [--label_modes given] [--out DIR] [--nsplit 5] [--tmax N]
Model per interval k (forward): z' = z + [global affine displacement] + class-wise affine residual (ridge, LOO-GCV, fit on the TRAIN half only; 5 stratified 50/50 splits).
  Rres_y : class-wise residual linear map (affine frame = global affine removed)           -> generator Lres_y = Rres_y / dstep   (displacement matrix, so logm(I+R)/dstep ~ R/dstep)
  Atot_y : A_glob + Rres_y  (total class-wise linear map)                                  -> Ltot_y = logm(Atot_y)/dstep, fallback (Atot_y - I)/dstep
  Lglob  : logm(A_glob)/dstep
Per interval and split: spectral abscissa (max Re), number of expanding directions (Re>0), complex pairs (|Im|/|lambda| > 0.1), leading gap (Re l1 - Re l2), eigenvector condition number
(non-normality), eigenvalue tracking by Hungarian matching on eigenvector overlap and the mean matched jump |dl|/(|l|+|l'|). Eigen-event detection lives in vfield_spectra_events.py.
Gauge: eigenvalues are invariant under a FIXED linear change of coordinates; the latents' gauge changes between snapshots, so Lres/Ltot eigenvalues are only approximately gauge free
(a time-dependent gauge adds a rotation/shear term to A_glob and, to first order, a common term to every A_y; differences between classes and the Lres eigenvalues are far less affected).
"""
import os, sys
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, json, time
import numpy as np
import torch
from scipy.linalg import logm
from scipy.optimize import linear_sum_assignment
from latentdyn.vfield import core as V
from latentdyn.vfield import decomp as VD

OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/spectra"
DT = torch.float64
CPLX_TOL = 0.1
METRICS = ["abscissa_med", "abscissa_max", "abscissa_min", "n_exp", "n_cplx", "gap12", "logcond", "jump", "trace_over_d"]


def class_fits(Z, ycond, classes, tr, dev):
    T, n, d = Z.shape; Z0, Z1 = Z[:-1], Z[1:]; D = Z1 - Z0; trt = torch.tensor(tr, device=dev)
    beta = VD.global_affine_fit(Z0, Z1, trt); Aglob = beta[:, :d, :]; f1 = D - VD.apply_affine_disp(Z0, beta)
    mu = Z0[:, trt].mean(1, keepdim=True); sig = ((Z0[:, trt] - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30; Zs = (Z0 - mu) / sig
    R = torch.full((T - 1, len(classes), d, d), float("nan"), dtype=Z.dtype, device=dev)
    for ci, c in enumerate(classes):
        itr = tr[ycond[tr] == c]
        if len(itr) < 4: continue
        _, _, A = V.ridge_fit(Zs[:, torch.tensor(itr, device=dev)], f1[:, torch.tensor(itr, device=dev)]); R[:, ci] = A / sig
    return Aglob.cpu().numpy(), R.cpu().numpy()


def logm_safe(A, ds):
    d = A.shape[0]
    try:
        if np.all(np.isfinite(A)) and np.linalg.cond(A) < 1e8:
            L = logm(A)
            if np.all(np.isfinite(L)) and np.abs(L.imag).max() < 1e-6 * (np.abs(L.real).max() + 1e-30): return L.real / ds, False
    except Exception: pass
    return (A - np.eye(d)) / ds, True


def eig_metrics(L, track=False):
    """L (K,C,d,d) real -> dict of (K,) series (class-aggregated) + per-class abscissa (K,C) + matched trajectories if track."""
    K, C, d, _ = L.shape; ok = np.isfinite(L).all((2, 3)); Lc = np.where(ok[:, :, None, None], L, np.eye(d)[None, None] * 0.0)
    w, v = np.linalg.eig(Lc); order = np.argsort(-w.real, axis=2); w = np.take_along_axis(w, order, 2); v = np.take_along_axis(v, order[:, :, None, :], 3)
    sv = np.linalg.svd(v, compute_uv=False); logcond = np.log(sv[..., 0] / np.maximum(sv[..., -1], 1e-300)); absc = w.real[:, :, 0]
    nexp = (w.real > 0).sum(2).astype(float); ncp = ((w.imag > 0) & (np.abs(w.imag) > CPLX_TOL * np.abs(w))).sum(2).astype(float); gap = w.real[:, :, 0] - w.real[:, :, 1]
    for a in (absc, logcond, nexp, ncp, gap): a[~ok] = np.nan
    out = dict(abscissa_cls=absc, n_exp_cls=nexp, n_cplx_cls=ncp, logcond_cls=logcond)
    # tracking (Hungarian on |v^H v'|), eigenvalues re-indexed along time
    jump = np.full((K - 1, C), np.nan); traj = np.zeros((K, C, d), complex) if track else None
    for c in range(C):
        if not ok[:, c].all(): continue
        vn = v[:, c] / np.maximum(np.linalg.norm(v[:, c], axis=1, keepdims=True), 1e-300); cur = w[0, c].copy(); prev_v = vn[0]
        if track: traj[0, c] = cur
        for k in range(K - 1):
            ov = np.abs(prev_v.conj().T @ vn[k + 1]); r, cidx = linear_sum_assignment(-ov); wn = w[k + 1, c][cidx]; vv = vn[k + 1][:, cidx]
            jump[k, c] = np.mean(np.abs(wn - cur) / (np.abs(wn) + np.abs(cur) + 1e-300)); cur, prev_v = wn, vv
            if track: traj[k + 1, c] = cur
    out["jump_cls"] = jump
    out["abscissa_med"] = np.nanmedian(absc, 1); out["abscissa_max"] = np.nanmax(np.where(np.isnan(absc), -np.inf, absc), 1); out["abscissa_min"] = np.nanmin(np.where(np.isnan(absc), np.inf, absc), 1)
    out["n_exp"] = np.nanmean(nexp, 1); out["n_cplx"] = np.nanmean(ncp, 1); out["gap12"] = np.nanmedian(gap, 1); out["logcond"] = np.nanmedian(logcond, 1)
    out["jump"] = np.concatenate([np.nanmean(jump, 1), [np.nan]]); out["trace_over_d"] = np.nanmedian(np.trace(Lc, axis1=2, axis2=3) / d, 1)
    if track: out["traj"] = traj
    return out


def global_gen(Aglob, ds):
    K, d, _ = Aglob.shape; Lg = np.zeros((K, d, d)); fb = np.zeros(K, bool)
    for k in range(K): Lg[k], fb[k] = logm_safe(Aglob[k].T, ds[k])
    ev = np.linalg.eigvals(Lg); return dict(abscissa=ev.real.max(1), n_exp=(ev.real > 0).sum(1).astype(float), n_cplx=((ev.imag > 0) & (np.abs(ev.imag) > CPLX_TOL * np.abs(ev))).sum(1).astype(float), trace_over_d=np.trace(Lg, axis1=1, axis2=2) / d), fb


def process(rd, probe, dev, nsplit, label_mode="given", tmax=0):
    Z, steps, ytrue, noisy = V.load_run(rd, probe)
    if tmax: Z, steps = Z[:tmax], steps[:tmax]
    ycond = (noisy if (noisy is not None and label_mode == "given") else ytrue); classes = np.unique(ycond); Zt = torch.tensor(Z, dtype=DT, device=dev); ds = np.diff(steps).astype(float); t0 = time.time()
    res = {m + "_res": [] for m in METRICS}; res.update({m + "_tot": [] for m in METRICS}); gl = collections_dd(); extra = {}
    for sp in range(nsplit):
        tr, te = V.make_split(ycond, sp); Ag, R = class_fits(Zt, ycond, classes, tr, dev)
        Lres = R / ds[:, None, None, None]; em = eig_metrics(Lres, track=(sp == 0))
        for m in METRICS: res[m + "_res"].append(em[m])
        if sp == 0:
            extra["traj_res_re"] = em["traj"].real.astype(np.float32); extra["traj_res_im"] = em["traj"].imag.astype(np.float32); extra["abscissa_cls_res"] = em["abscissa_cls"].astype(np.float32); extra["n_exp_cls_res"] = em["n_exp_cls"].astype(np.float32)
        K, C, d, _ = R.shape; Ltot = np.zeros_like(R); fbt = 0
        for k in range(K):
            for c in range(C):
                Ltot[k, c], fb = logm_safe(Ag[k] + R[k, c], ds[k]); fbt += fb
        Atot = Ltot; emt = eig_metrics(Ltot, track=False)
        for m in METRICS: res[m + "_tot"].append(emt[m])
        g, fb = global_gen(Ag, ds)
        for k_, v_ in g.items(): gl[k_].append(v_)
        gl["fallback"].append(np.array([fb.mean(), fbt / (K * C)]))
    out = {k: np.array(v, np.float32) for k, v in res.items()}
    for k, v in gl.items(): out["glob_" + k] = np.array(v, np.float32)
    out.update(extra); out["steps"] = steps; out["n"] = np.array(Z.shape[1]); out["min_class"] = np.array(min(int((ycond == c).sum()) for c in classes)); out["seconds"] = np.array(time.time() - t0)
    return out


def collections_dd():
    import collections; return collections.defaultdict(list)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); r = sp.add_parser("run")
    r.add_argument("--device", default="cpu"); r.add_argument("--runs", required=True); r.add_argument("--probes", default="test"); r.add_argument("--label_modes", default="given"); r.add_argument("--out", default=OUT)
    r.add_argument("--nsplit", type=int, default=5); r.add_argument("--limit", type=int, default=0); r.add_argument("--force", action="store_true"); r.add_argument("--tmax", type=int, default=0)
    a = ap.parse_args(); dev = torch.device(a.device); torch.set_num_threads(8)
    assert dev.type == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES", "") == "3", "GPU runs must be pinned to GPU 3 (CUDA_VISIBLE_DEVICES=3)"
    os.makedirs(f"{a.out}/runs", exist_ok=True)
    dirs = sorted(d for g in a.runs.split(",") for d in glob.glob(f"{V.RUNS}/{g}") if os.path.isdir(d) and os.path.isfile(f"{d}/model_final.pt"))
    if a.limit: dirs = dirs[:a.limit]
    for d in dirs:
        rid = os.path.basename(d)
        for probe in a.probes.split(","):
            for lm in a.label_modes.split(","):
                if probe == "test" and lm == "true": continue
                if probe == "train" and not os.path.isdir(f"{d}/latents_train"): continue
                fo = f"{a.out}/runs/{rid}_{probe}_{lm}.npz"
                if os.path.isfile(fo) and not a.force: continue
                res = process(d, probe, dev, a.nsplit, lm, a.tmax); np.savez_compressed(fo, **res); print(f"{rid} {probe} {lm} done {float(res['seconds']):.0f}s", flush=True)
                if dev.type == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__": main()
