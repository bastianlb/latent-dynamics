"""Train-probe vs test-probe latent dynamics: field transfer, contraction, label-noise drag, dense per-step lags.   torch (GPU optional), numpy for eigenvalues.
Usage:
  python vfield_traintest.py run   --device cuda --runs 'GLOB,...' [--nsplit 5] [--out DIR] [--force]
  python vfield_traintest.py dense --device cuda --dense_runs dl_adam_n4000_wd0,dl_adam_n4000_ln0.2 [--out DIR]
Setup per interval k (snapshots t_k -> t_{k+1}), displacement d_i = z_{t+1,i} - z_{t,i}, forward global affine A_t z + b_t removed (affine frame), class-wise affine ridge on the residual
(vfield.ridge_fit, LOO-GCV).  Sets: TEST probe (1000, true labels) and TRAIN probe (min(N,1000) points, given labels = the labels trained on).  5 stratified 50/50 splits per set.
Transfer: fit on the fit-half of the source set, score uncentered R2 (fraction of sum ||d||^2) on the held-out half of the target set.
  global mode 'sep': global affine fitted on the source set;  'pool': fitted on the pooled fit-halves of both sets.  Class-wise ridge is always fitted on the source set.
  R2g = global affine only, R2c = global + class-wise.  Transfer gap(target) = R2(within: source=target) - R2(cross: other set).
"""
import os, sys
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, json, time
import numpy as np
import torch
from latentdyn.vfield import core as V
from latentdyn.vfield import decomp as VD

OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/traintest"
DT = torch.float64
SETS = ["te", "tr"]
GM = ["sep", "pool"]


def load_scalars(rd):
    fs = sorted(glob.glob(f"{rd}/latents/step_*.npz")); out = {k: [] for k in ("step", "train_acc", "test_acc", "train_loss", "test_loss")}
    for f in fs:
        with np.load(f) as z:
            for k in out: out[k].append(float(z[k]) if k in z.files else np.nan)
    o = np.argsort(out["step"]); return {k: np.array(v)[o] for k, v in out.items()}


def ridge_class_fit(Zs, F, idx_cls, classes, ycl):
    """per-class ridge on fit indices -> dict class -> (xb, db, A)."""
    fits = {}
    for c in classes:
        itr = idx_cls[c]
        if len(itr) < 4: continue
        fits[c] = V.ridge_fit(Zs[:, itr], F[:, itr])
    return fits


def predict_class(fits, Zs_t, y_t, H, classes, dev):
    K, n, d = Zs_t.shape; out = torch.zeros(K, len(H), d, dtype=Zs_t.dtype, device=dev); pos = {int(i): j for j, i in enumerate(H)}
    for c in classes:
        if c not in fits: continue
        ih = H[y_t[H] == c]
        if len(ih) == 0: continue
        sel = torch.tensor([pos[int(i)] for i in ih], device=dev); xb, db, A = fits[c]; out[:, sel] = db[:, None] + (Zs_t[:, torch.tensor(ih, device=dev)] - xb[:, None]) @ A
    return out


def r2u(pred, target):
    return 1 - ((target - pred) ** 2).sum((1, 2)) / ((target ** 2).sum((1, 2)) + 1e-300)


def transfer_block(S, nsplit, dev, classes):
    """S: dict set -> dict(Z0, Z1, y); returns array (nsplit, 2 gm, 2 src, 2 tgt, 2 (g,c), K)."""
    K = S["te"]["Z0"].shape[0]; out = np.full((nsplit, 2, 2, 2, 2, K), np.nan, np.float32)
    for sp in range(nsplit):
        sub = {}
        for s in SETS:
            F, H = V.make_split(S[s]["y"], sp); sub[s] = dict(F=F, H=H, Ft=torch.tensor(F, device=dev), Ht=torch.tensor(H, device=dev)); sub[s]["D"] = S[s]["Z1"] - S[s]["Z0"]
        beta = {s: VD.global_affine_fit(S[s]["Z0"], S[s]["Z1"], sub[s]["Ft"]) for s in SETS}
        Zp0 = torch.cat([S["te"]["Z0"][:, sub["te"]["Ft"]], S["tr"]["Z0"][:, sub["tr"]["Ft"]]], 1); Zp1 = torch.cat([S["te"]["Z1"][:, sub["te"]["Ft"]], S["tr"]["Z1"][:, sub["tr"]["Ft"]]], 1)
        beta_pool = VD.global_affine_fit(Zp0, Zp1, torch.arange(Zp0.shape[1], device=dev))
        for gi, gm in enumerate(GM):
            for si, s in enumerate(SETS):
                b = beta[s] if gm == "sep" else beta_pool; Z0 = S[s]["Z0"]; f = sub[s]["D"] - VD.apply_affine_disp(Z0, b)
                Ft = sub[s]["Ft"]; mu = Z0[:, Ft].mean(1, keepdim=True); sig = ((Z0[:, Ft] - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30; Zs = (Z0 - mu) / sig
                y = S[s]["y"]; idx_cls = {c: torch.tensor(sub[s]["F"][y[sub[s]["F"]] == c], device=dev) for c in classes}
                fits = ridge_class_fit(Zs, f, idx_cls, classes, y)
                for ti, t in enumerate(SETS):
                    H = sub[t]["H"]; Ht = sub[t]["Ht"]; Z0t = S[t]["Z0"]; Dt = sub[t]["D"][:, Ht]
                    p1 = VD.apply_affine_disp(Z0t, b)[:, Ht]; Zs_t = (Z0t - mu) / sig; p2 = predict_class(fits, Zs_t, S[t]["y"], H, classes, dev)
                    out[sp, gi, si, ti, 0] = r2u(p1, Dt).cpu().numpy(); out[sp, gi, si, ti, 1] = r2u(p1 + p2, Dt).cpu().numpy()
    return out


def full_class_fits(Z0, D, y, classes, beta, dev, mu_sig=None):
    """class-wise ridge generator Rraw (K,C,d,d) (raw coordinates) and class drift db (K,C,d) on ALL points of a set after global affine beta."""
    K, n, d = Z0.shape; f = D - VD.apply_affine_disp(Z0, beta)
    if mu_sig is None: mu = Z0.mean(1, keepdim=True); sig = ((Z0 - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30
    else: mu, sig = mu_sig
    Zs = (Z0 - mu) / sig; R = torch.full((K, len(classes), d, d), float("nan"), dtype=Z0.dtype, device=dev); M = R[:, :, 0].clone()
    for ci, c in enumerate(classes):
        ii = torch.tensor(np.where(y == c)[0], device=dev)
        if len(ii) < 4: continue
        xb, db, A = V.ridge_fit(Zs[:, ii], f[:, ii]); R[:, ci] = A / sig; M[:, ci] = db
    return R, M


def rel_dist(a, b):
    na, nb = np.linalg.norm(a.reshape(a.shape[:2] + (-1,)), axis=2), np.linalg.norm(b.reshape(b.shape[:2] + (-1,)), axis=2)
    return np.linalg.norm((a - b).reshape(a.shape[:2] + (-1,)), axis=2) / (0.5 * (na + nb) + 1e-300)


def abscissa_div(R, ds):
    L = R / ds[:, None, None, None]; K, C, d, _ = L.shape; ok = np.isfinite(L).all((2, 3)); Lc = np.where(ok[:, :, None, None], L, 0.0); w = np.linalg.eigvals(Lc); ab = w.real.max(2); ab[~ok] = np.nan
    dv = np.trace(Lc, axis1=2, axis2=3); dv[~ok] = np.nan; return np.nanmedian(ab, 1), np.nanmean(dv, 1)


def trace_ratio(Z, y, classes):
    """tr(Sw)/tr(Sb) per snapshot, torch (T,n,d)."""
    T, n, d = Z.shape; g = Z.mean(1); sw = torch.zeros(T, dtype=Z.dtype, device=Z.device); sb = sw.clone()
    for c in classes:
        ii = torch.tensor(np.where(y == c)[0], device=Z.device); zc = Z[:, ii]; mc = zc.mean(1); p = len(ii) / n
        sw = sw + p * ((zc - mc[:, None]) ** 2).sum(2).mean(1); sb = sb + p * ((mc - g) ** 2).sum(1)
    return (sw / sb).cpu().numpy()


def ncm_acc(Z, y, classes):
    T, n, d = Z.shape; mus = torch.stack([Z[:, torch.tensor(np.where(y == c)[0], device=Z.device)].mean(1) for c in classes], 1); dist = torch.cdist(Z, mus); pred = dist.argmin(2).cpu().numpy()
    cm = np.array([np.where(classes == c)[0][0] for c in y]); return (pred == cm[None, :]).mean(1)


def drag(Zte, Ztr, yte, ytrue, ygiven, beta_pool, dev, classes):
    """label-noise drag: cos of test-class drift toward mislabelled-cluster mean vs clean mean; per interval, class-averaged."""
    K = Zte.shape[0] - 1; bad = ytrue != ygiven; clean = ~bad; out = {}
    fte = (Zte[1:] - Zte[:-1]) - VD.apply_affine_disp(Zte[:-1], beta_pool); ftr = (Ztr[1:] - Ztr[:-1]) - VD.apply_affine_disp(Ztr[:-1], beta_pool); z0te, z0tr = Zte[:-1], Ztr[:-1]
    res = {k: np.full((K, len(classes)), np.nan) for k in ("cos_toM", "cos_toC", "cos_toRand", "cosM_drift_toC", "cosM_drift_toTrue", "nM")}; rng = np.random.RandomState(0)
    for ci, c in enumerate(classes):
        iT = np.where(yte == c)[0]; iC = np.where(clean & (ygiven == c))[0]; iM = np.where(bad & (ygiven == c))[0]
        if len(iM) < 5 or len(iC) < 5 or len(iT) < 5: continue
        iR = rng.choice(iC, size=min(len(iM), len(iC)), replace=False); tT, tC, tM, tR = [torch.tensor(i, device=dev) for i in (iT, iC, iM, iR)]
        mT, mC, mM, mR = z0te[:, tT].mean(1), z0tr[:, tC].mean(1), z0tr[:, tM].mean(1), z0tr[:, tR].mean(1); dT = fte[:, tT].mean(1); dM = ftr[:, tM].mean(1)
        cs = lambda a, b: ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-300)).cpu().numpy()
        res["cos_toM"][:, ci] = cs(dT, mM - mT); res["cos_toC"][:, ci] = cs(dT, mC - mT); res["cos_toRand"][:, ci] = cs(dT, mR - mT); res["cosM_drift_toC"][:, ci] = cs(dM, mC - mM); res["nM"][:, ci] = len(iM)
        # true class of the mislabelled points: mean position of clean train points of each true class
        tc = ytrue[iM]; mt = torch.stack([z0tr[:, torch.tensor(np.where(clean & (ygiven == t))[0], device=dev)].mean(1) for t in tc], 1)   # (K,|M|,d)
        v = mt - z0tr[:, tM]; dd = ftr[:, tM]; res["cosM_drift_toTrue"][:, ci] = ((dd * v).sum(2) / (dd.norm(dim=2) * v.norm(dim=2) + 1e-300)).mean(1).cpu().numpy()
    return {k: np.nanmean(v, 1).astype(np.float32) for k, v in res.items()}


def process(rd, dev, nsplit, tmax=0):
    Zte, steps, yte, _ = V.load_run(rd, "test"); Ztr, steps2, ytrue, noisy = V.load_run(rd, "train"); sc = load_scalars(rd)
    if tmax: Zte, Ztr, steps, steps2 = Zte[:tmax], Ztr[:tmax], steps[:tmax], steps2[:tmax]
    ygiven = noisy if noisy is not None else ytrue; classes = np.unique(yte); t0 = time.time(); ds = np.diff(steps).astype(float)
    Te = torch.tensor(Zte, dtype=DT, device=dev); Tr = torch.tensor(Ztr, dtype=DT, device=dev)
    S = {"te": dict(Z0=Te[:-1], Z1=Te[1:], y=yte), "tr": dict(Z0=Tr[:-1], Z1=Tr[1:], y=ygiven)}
    out = dict(steps=steps, n_te=np.array(Zte.shape[1]), n_tr=np.array(Ztr.shape[1]), min_class_tr=np.array(min(int((ygiven == c).sum()) for c in classes)))
    T_ = transfer_block(S, nsplit, dev, classes); out["transfer_mean"] = np.nanmean(T_, 0); out["transfer_std"] = np.nanstd(T_, 0)         # (2gm, 2src, 2tgt, 2, K)
    # direct differences (split-0 halves, pooled global on all points)
    Zp0 = torch.cat([S["te"]["Z0"], S["tr"]["Z0"]], 1); Zp1 = torch.cat([S["te"]["Z1"], S["tr"]["Z1"]], 1); beta_pool = VD.global_affine_fit(Zp0, Zp1, torch.arange(Zp0.shape[1], device=dev))
    mu = Zp0.mean(1, keepdim=True); sig = ((Zp0 - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30
    R_h = {}; M_h = {}
    for s in SETS:
        F, H = V.make_split(S[s]["y"], 0)
        for hn, idx in (("A", F), ("B", H)):
            sub_y = S[s]["y"][idx]; Z0h = S[s]["Z0"][:, torch.tensor(idx, device=dev)]; Dh = (S[s]["Z1"] - S[s]["Z0"])[:, torch.tensor(idx, device=dev)]
            R, M = full_class_fits(Z0h, Dh, sub_y, classes, beta_pool, dev, mu_sig=(mu, sig)); R_h[(s, hn)] = R.cpu().numpy(); M_h[(s, hn)] = M.cpu().numpy()
    def dists(Hd):
        w = lambda s: rel_dist(Hd[(s, "A")], Hd[(s, "B")]); cross = np.mean([rel_dist(Hd[("te", a)], Hd[("tr", b)]) for a in "AB" for b in "AB"], 0); return w("te"), w("tr"), cross
    wa, wb, ca = dists(R_h); out["A_rel_dist_within_te"] = np.nanmedian(wa, 1); out["A_rel_dist_within_tr"] = np.nanmedian(wb, 1); out["A_rel_dist_cross"] = np.nanmedian(ca, 1)
    wa, wb, ca = dists(M_h); out["drift_rel_dist_within_te"] = np.nanmedian(wa, 1); out["drift_rel_dist_within_tr"] = np.nanmedian(wb, 1); out["drift_rel_dist_cross"] = np.nanmedian(ca, 1)
    # contraction: separate fits on each set (full sets, own global affine)
    for s, Tt in (("te", Te), ("tr", Tr)):
        beta_s = VD.global_affine_fit(Tt[:-1], Tt[1:], torch.arange(Tt.shape[1], device=dev)); R, _ = full_class_fits(Tt[:-1], Tt[1:] - Tt[:-1], S[s]["y"], classes, beta_s, dev); ab, dv = abscissa_div(R.cpu().numpy(), ds)
        out[f"abscissa_{s}"] = ab; out[f"divergence_{s}"] = dv
    out["tr_ratio_te"] = trace_ratio(Te, yte, classes); out["tr_ratio_tr_true"] = trace_ratio(Tr, ytrue, classes); out["tr_ratio_tr_given"] = trace_ratio(Tr, ygiven, classes)
    out["ncm_te"] = ncm_acc(Te, yte, classes); out["ncm_tr_true"] = ncm_acc(Tr, ytrue, classes)
    for k in ("train_acc", "test_acc", "train_loss", "test_loss"): out[f"snap_{k}"] = sc[k][:len(steps)]
    if noisy is not None and (noisy != ytrue).any():
        dg = drag(Te, Tr, yte, ytrue, ygiven, beta_pool, dev, classes)
        for k, v in dg.items(): out[f"drag_{k}"] = v
    out["seconds"] = np.array(time.time() - t0); return out


# ----------------------------------------------------------------------------- dense (per-step lags, fp16 floor, logit frame)
def ulp16(x):
    """fp16 spacing at |x| (normal range); x float32 tensor."""
    e = torch.floor(torch.log2(x.abs().clamp_min(2 ** -14))); return torch.pow(2.0, e - 10)


def load_dense(dr, dev):
    meta = json.load(open(f"{dr}/meta.json")); chunks = sorted(glob.glob(f"{dr}/chunk_*.npz")); Zte, Ztr, Wh, bh, steps = [], [], [], [], []
    for c in chunks:
        with np.load(c) as z:
            Zte.append(torch.tensor(z["latents_test"], device=dev)); Ztr.append(torch.tensor(z["latents_train"], device=dev)); Wh.append(torch.tensor(z["head_w"], device=dev)); bh.append(torch.tensor(z["head_b"], device=dev)); steps.append(z["steps"])
    return torch.cat(Zte), torch.cat(Ztr), torch.cat(Wh), torch.cat(bh), np.concatenate(steps)


def dense_run(dname, dev, nsplit, out, tgrid=36, lags=(1, 2, 5, 10, 30, 100, 300, 1000)):
    dr = f"/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent/mnist_probe_none_d16_s0_{dname}"; art = f"/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent/_run_artifacts/runs/mnist_probe_none_d16_s0_{dname}"
    Zte16, Ztr16, Wh, bh, steps = load_dense(dr, dev); cfg = json.load(open(f"{art}/config.json")); pi = np.load(f"{dr}/probe_info.npz"); ygiven = pi["train_probe_labels_used"].astype(np.int64)
    with np.load(f"{art}/probe_index.npz") as z: yte = z["labels"].astype(np.int64)
    ytrue = ygiven.copy()
    try:
        with np.load(f"{art}/probe_train_index.npz") as z: ytrue = z["labels"].astype(np.int64)
    except Exception: pass
    classes = np.unique(yte); T = len(steps); assert steps[0] == 0 and steps[-1] == T - 1, "expects every step logged"
    ts = np.unique(np.round(np.geomspace(5, T - 2, tgrid)).astype(int)); rows = []; cells = []
    for L in lags:
        for t in ts:
            if t + L <= T - 1: cells.append((t, L))
    K = len(cells); ti = torch.tensor([c[0] for c in cells], device=dev); tj = torch.tensor([c[0] + c[1] for c in cells], device=dev)
    res = {}
    for frame in ("latent", "logit"):
        if frame == "latent":
            Z0 = {"te": Zte16[ti].double(), "tr": Ztr16[ti].double()}; Z1 = {"te": Zte16[tj].double(), "tr": Ztr16[tj].double()}
            ulp0 = {"te": ulp16(Zte16[ti].float()), "tr": ulp16(Ztr16[ti].float())}; ulp1 = {"te": ulp16(Zte16[tj].float()), "tr": ulp16(Ztr16[tj].float())}
            floor = {s: ((ulp0[s].double() ** 2 + ulp1[s].double() ** 2) / 12).sum((1, 2)) for s in SETS}      # total floor energy per cell (all points)
        else:
            Lg = lambda Z, idx: torch.einsum("knd,kcd->knc", Z.float(), Wh[idx]) + bh[idx][:, None, :]
            Z0 = {"te": Lg(Zte16[ti], ti).double(), "tr": Lg(Ztr16[ti], ti).double()}; Z1 = {"te": Lg(Zte16[tj], tj).double(), "tr": Lg(Ztr16[tj], tj).double()}
            floor = {}   # floor on logits = sum_k ulp_k^2/12 * W_ck^2 (propagated fp16 rounding of z, both snapshots)
            for s, Z16 in (("te", Zte16), ("tr", Ztr16)):
                e0 = torch.einsum("knd,kcd->knc", ulp16(Z16[ti].float()) ** 2, Wh[ti].float() ** 2).double() / 12; e1 = torch.einsum("knd,kcd->knc", ulp16(Z16[tj].float()) ** 2, Wh[tj].float() ** 2).double() / 12; floor[s] = (e0 + e1).sum((1, 2))
        S = {"te": dict(Z0=Z0["te"], Z1=Z1["te"], y=yte), "tr": dict(Z0=Z0["tr"], Z1=Z1["tr"], y=ygiven)}
        cl = classes
        T_ = transfer_block(S, nsplit, dev, cl); res[f"{frame}_transfer"] = np.nanmean(T_, 0)
        # residual energy of the within-set class-affine model per point, to compare with the floor per point
        sse = {}
        for s, si in (("te", 1), ("tr", 0)):
            Dtot = (S[s]["Z1"] - S[s]["Z0"]); r2c = res[f"{frame}_transfer"][0, si, si, 1]; sst = (Dtot ** 2).sum((1, 2)).cpu().numpy(); sse[s] = (1 - r2c) * sst / Dtot.shape[1]
        res[f"{frame}_floor_ratio_te"] = floor["te"].cpu().numpy() / S["te"]["Z0"].shape[1] / np.maximum(sse["te"], 1e-300); res[f"{frame}_floor_ratio_tr"] = floor["tr"].cpu().numpy() / S["tr"]["Z0"].shape[1] / np.maximum(sse["tr"], 1e-300)
    np.savez_compressed(f"{out}/dense_{dname}.npz", cells=np.array(cells), steps=steps, **res, ncm_te=ncm_acc(Zte16.double(), yte, classes)[::50], steps_ncm=steps[::50])
    print("dense", dname, "cells", K, flush=True)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    r = sp.add_parser("run"); r.add_argument("--device", default="cpu"); r.add_argument("--runs", required=True); r.add_argument("--out", default=OUT); r.add_argument("--nsplit", type=int, default=5); r.add_argument("--limit", type=int, default=0); r.add_argument("--force", action="store_true"); r.add_argument("--tmax", type=int, default=0)
    d = sp.add_parser("dense"); d.add_argument("--device", default="cpu"); d.add_argument("--dense_runs", required=True); d.add_argument("--out", default=OUT); d.add_argument("--nsplit", type=int, default=5)
    a = ap.parse_args(); dev = torch.device(a.device); torch.set_num_threads(8)
    assert dev.type == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("1", "1,3", "1,4", "1,5", "1,6", "1,7"), "GPU runs must be pinned to GPU 1 (never GPU 0)"
    os.makedirs(f"{a.out}/runs", exist_ok=True)
    if a.cmd == "run":
        dirs = sorted(dd for g in a.runs.split(",") for dd in glob.glob(f"{V.RUNS}/{g}") if os.path.isdir(dd) and os.path.isfile(f"{dd}/model_final.pt") and os.path.isdir(f"{dd}/latents_train"))
        if a.limit: dirs = dirs[:a.limit]
        for dd in dirs:
            rid = os.path.basename(dd); fo = f"{a.out}/runs/{rid}.npz"
            if os.path.isfile(fo) and not a.force: continue
            res = process(dd, dev, a.nsplit, a.tmax); np.savez_compressed(fo, **res); print(f"{rid} done {float(res['seconds']):.0f}s", flush=True)
            if dev.type == "cuda": torch.cuda.empty_cache()
    else:
        for dn in a.dense_runs.split(","): dense_run(dn, dev, a.nsplit, a.out)


if __name__ == "__main__": main()
