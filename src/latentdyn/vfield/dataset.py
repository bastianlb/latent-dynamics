"""Replicate the latent-dynamics analyses (partition, spectra, train-vs-test transfer/contraction/drag, mislabelled flip + memorisation crossover) on other datasets.
Usage: python vfield_dataset.py run --dataset fm|cifar|mnist_ref --runs 'GLOB,...' --out ROOT [--stages traintest,spectra,partition,flip,null] [--proj none|pca16|pca32|logit10] [--device cpu]
Reuses vfield.py / vfield_decomp.py / vfield_spectra.py / vfield_traintest.py unchanged (except the optional `light` partition mode and a guarded final-head block) by
monkey-patching vfield.load_run, so high-dimensional latents are first projected to a FIXED run-level linear frame:
  pca<k> : orthonormal top-k principal directions of the pooled, per-snapshot-normalised (centred, divided by RMS radius), test+train probe latents of the run (every 2nd snapshot);
  logit10: logits = z W^T + b with the FINAL probe head from model_final.pt (fixed late-time readout; not the head of earlier snapshots).
The fraction of displacement energy inside the projected subspace is logged to ROOT/proj_info.csv.
"""
import os, sys
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, json, time, csv
import numpy as np
import torch
from latentdyn.vfield import core as V
from latentdyn.vfield import decomp as VD
from latentdyn.vfield import spectra as VS
from latentdyn.vfield import traintest as VT

ORIG_LOAD = V.load_run


def is_noisy(rid):
    import re
    return "_gap_" in rid or re.search(r"_ln0\.[1-9]\d*$", rid) is not None
PROJ = dict(mode="none", k=0)
BASIS = {}
DT = torch.float64


def basis_for(rd, k):
    key = (rd, k)
    if key in BASIS: return BASIS[key]
    rows = []
    for probe in ("test", "train"):
        Z, steps, y, noisy = ORIG_LOAD(rd, probe)
        for t in range(0, len(Z), 2):
            z = Z[t] - Z[t].mean(0, keepdims=True); rows.append((z / (np.sqrt((z ** 2).sum(1).mean()) + 1e-30)).astype(np.float32))
    X = np.concatenate(rows, 0); C = (X.T @ X) / len(X); w, v = np.linalg.eigh(C); P = v[:, ::-1][:, :k].copy(); BASIS[key] = (P, float(w[::-1][:k].sum() / w.sum())); return BASIS[key]


def patched_load(rd, probe):
    Z, steps, y, noisy = ORIG_LOAD(rd, probe); mode = PROJ["mode"]
    if mode == "none": return Z, steps, y, noisy
    if mode == "logit10":
        sd = torch.load(f"{rd}/model_final.pt", map_location="cpu", weights_only=False); W = sd["probe.weight"].double().numpy(); b = sd["probe.bias"].double().numpy(); return Z @ W.T + b, steps, y, noisy
    P, _ = basis_for(rd, PROJ["k"]); return Z @ P, steps, y, noisy


def proj_info(rd, out):
    if PROJ["mode"] == "none": return
    Z, steps, y, n = ORIG_LOAD(rd, "test"); Zp = patched_load(rd, "test")[0]; D = Z[1:] - Z[:-1]; Dp = Zp[1:] - Zp[:-1]
    if PROJ["mode"].startswith("pca"):
        P, var = basis_for(rd, PROJ["k"]); inside = (((D @ P) ** 2).sum((1, 2)) / (D ** 2).sum((1, 2)))
    else: var = np.nan; inside = np.full(len(D), np.nan)
    row = dict(run_id=os.path.basename(rd), proj=PROJ["mode"], latent_variance_in_basis=var, disp_energy_in_subspace_median=float(np.nanmedian(inside)), disp_energy_in_subspace_late=float(np.nanmedian(inside[len(inside) // 2:])))
    new = not os.path.isfile(f"{out}/proj_info.csv")
    with open(f"{out}/proj_info.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); (w.writeheader() if new else None); w.writerow(row)


def mem_cross_perclass(Z, steps, ytrue, noisy):
    """per-class persistent crossover of NCM(noisy-label means) fitting noisy vs true label on corrupted items (as perclass.py), median over classes."""
    classes = np.unique(ytrue); bad = noisy != ytrue; T = len(Z); fit = np.full((T, 10), np.nan); cor = fit.copy()
    for t in range(T):
        mu = np.array([Z[t][noisy == c].mean(0) if (noisy == c).any() else np.full(Z.shape[2], 1e9) for c in classes]); pred = classes[((Z[t][:, None, :] - mu[None]) ** 2).sum(-1).argmin(1)]
        for ci, c in enumerate(classes):
            m = bad & (ytrue == c)
            if m.sum() >= 3: fit[t, ci] = (pred[m] == noisy[m]).mean(); cor[t, ci] = (pred[m] == ytrue[m]).mean()
    out = []
    for ci in range(len(classes)):
        ok = np.nan_to_num(fit[:, ci]) > np.nan_to_num(cor[:, ci])
        if not ok[-1]: continue
        b_ = np.where(~ok)[0]; out.append(float(steps[0 if len(b_) == 0 else b_[-1] + 1]))
    return float(np.median(out)) if out else np.nan, fit, cor


def persistent_cross(diff, steps, frac=0.8):
    for k in range(len(diff)):
        if diff[k] > 0 and np.nanmean(diff[k:] > 0) >= frac: return float(steps[k])
    return np.nan


def stage_flip(rd, dev, out):
    Zr, steps, ytrue, noisy = ORIG_LOAD(rd, "train")
    if noisy is None or not (noisy != ytrue).any(): return None
    Zt = torch.tensor(patched_load(rd, "train")[0], dtype=DT, device=dev); ml = V.mislabeled(Zt, ytrue, noisy, dev); mem, fit, cor = mem_cross_perclass(Zr, steps, ytrue, noisy)
    # memorisation crossover is computed in the ORIGINAL latent space (NCM on class means), flip in the projected frame
    res = dict(steps=steps, n_bad=np.array(ml.get("n_bad", 0)), mem_cross_step=np.array(mem))
    for f in V_FRAMES:
        if f in ml:
            for k, v in ml[f].items(): res[f"mis_{f}_{k}"] = v
            d = ml[f]["cos_given"] - ml[f]["cos_true"]; res[f"flip_step_{f}"] = np.array(persistent_cross(d, steps[1:]))
    return res


V_FRAMES = ["raw", "procrustes", "affine"]


def stage_null(rd, dev, nsplit=1):
    """leading eigenvalue of the class-wise residual generator vs the shuffled-displacement null (test probe, split 0)."""
    Z, steps, y, _ = V.load_run(rd, "test"); classes = np.unique(y); Zt = torch.tensor(Z, dtype=DT, device=dev); ds = np.diff(steps).astype(float); tr, te = V.make_split(y, 0)
    Ag, R = VS.class_fits(Zt, y, classes, tr, dev)
    T, n, d = Z.shape; Z0, Z1 = Zt[:-1], Zt[1:]; D = Z1 - Z0; trt = torch.tensor(tr, device=dev); beta = VD.global_affine_fit(Z0, Z1, trt); f1 = D - VD.apply_affine_disp(Z0, beta)
    mu = Z0[:, trt].mean(1, keepdim=True); sig = ((Z0[:, trt] - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30; Zs = (Z0 - mu) / sig; Rn = np.zeros_like(R); rng = np.random.RandomState(0)
    for ci, c in enumerate(classes):
        itr = tr[y[tr] == c]; perm = torch.tensor(rng.permutation(len(itr)), device=dev); _, _, A = V.ridge_fit(Zs[:, torch.tensor(itr, device=dev)], f1[:, torch.tensor(itr, device=dev)][:, perm]); Rn[:, ci] = (A / sig).cpu().numpy()
    def stats(M):
        w = np.linalg.eigvals(M / ds[:, None, None, None]); return w.real.max(2), (w.real > 0).sum(2).astype(float), ((w.imag > 0) & (np.abs(w.imag) > 0.1 * np.abs(w))).sum(2).astype(float)
    a, e, c_ = stats(R); an, en, cn = stats(Rn)
    return dict(steps=steps, abscissa_data=np.nanmedian(a, 1), abscissa_null=np.nanmedian(an, 1), n_exp_data=e.mean(1), n_exp_null=en.mean(1), n_cplx_data=c_.mean(1), n_cplx_null=cn.mean(1), abscissa_cls_data=a)


def run_one(rd, root, stages, dev, nsplit_part=2):
    rid = os.path.basename(rd); has_train = os.path.isdir(f"{rd}/latents_train")
    for st in stages:
        od = f"{root}/{st}/runs"; os.makedirs(od, exist_ok=True); t0 = time.time()
        if st == "traintest":
            fo = f"{od}/{rid}.npz"
            if not os.path.isfile(fo) and has_train:
                res = VT.process(rd, dev, 5)
                if PROJ["mode"] != "none":   # contraction (trace ratio) and NCM accuracy in the ORIGINAL latent space as well
                    Zte, _, yte, _ = ORIG_LOAD(rd, "test"); Ztr, _, ytrue, noisy = ORIG_LOAD(rd, "train"); cl = np.unique(yte)
                    res["tr_ratio_te_full"] = VT.trace_ratio(torch.tensor(Zte, dtype=DT), yte, cl); res["tr_ratio_tr_true_full"] = VT.trace_ratio(torch.tensor(Ztr, dtype=DT), ytrue, cl)
                    res["ncm_te_full"] = VT.ncm_acc(torch.tensor(Zte, dtype=DT), yte, cl); res["ncm_tr_true_full"] = VT.ncm_acc(torch.tensor(Ztr, dtype=DT), ytrue, cl)
                np.savez_compressed(fo, **res)
        elif st == "spectra":
            probes = [("test", "given")] + ([("train", "given"), ("train", "true")] if (has_train and is_noisy(rid)) else [])
            for probe, lm in probes:
                fo = f"{od}/{rid}_{probe}_{lm}.npz"
                if not os.path.isfile(fo): np.savez_compressed(fo, **VS.process(rd, probe, dev, 5, lm))
        elif st == "partition":
            for probe in ("test", "train") if has_train else ("test",):
                fo = f"{od}/{rid}_{probe}.npz"
                if os.path.isfile(fo): continue
                Z, steps, ytrue, noisy = V.load_run(rd, probe); ycond = noisy if noisy is not None else ytrue; classes = np.unique(ycond)
                names, P = VD.partition(torch.tensor(Z, dtype=DT, device=dev), steps, ycond, classes, dev, nsplit_part, light=True)
                np.savez_compressed(fo, steps=steps, part_names=np.array(names), part_mean=P.mean(0), part_std=P.std(0), n=np.array(Z.shape[1]), min_class=np.array(min(int((ycond == c).sum()) for c in classes)))
        elif st == "flip":
            fo = f"{od}/{rid}.npz"
            if is_noisy(rid) and has_train and not os.path.isfile(fo):
                r = stage_flip(rd, dev, root)
                if r is not None: np.savez_compressed(fo, **r)
        elif st == "null":
            fo = f"{od}/{rid}.npz"
            if not os.path.isfile(fo): np.savez_compressed(fo, **stage_null(rd, dev))
        print(f"{rid} {st} {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); r = sp.add_parser("run")
    r.add_argument("--runs", required=True); r.add_argument("--out", required=True); r.add_argument("--stages", default="traintest,spectra,partition,flip,null"); r.add_argument("--proj", default="none"); r.add_argument("--device", default="cpu"); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--nsplit_part", type=int, default=2)
    a = ap.parse_args(); dev = torch.device(a.device); torch.set_num_threads(8)
    assert dev.type == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "0"), "pin an idle non-zero GPU"
    PROJ["mode"] = "none" if a.proj == "none" else ("logit10" if a.proj == "logit10" else "pca"); PROJ["k"] = int(a.proj[3:]) if a.proj.startswith("pca") else 0
    if PROJ["mode"] == "pca": PROJ["mode"] = "pca"
    V.load_run = patched_load
    dirs = sorted(d for g in a.runs.split(",") for d in glob.glob(f"{V.RUNS}/{g}") if os.path.isdir(d) and os.path.isfile(f"{d}/model_final.pt"))
    if a.limit: dirs = dirs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    for d in dirs:
        if a.proj != "none": proj_info(d, a.out) if not os.path.isfile(f"{a.out}/.pi_{os.path.basename(d)}") else None; open(f"{a.out}/.pi_{os.path.basename(d)}", "w").close()
        run_one(d, a.out, a.stages.split(","), dev, a.nsplit_part)


if __name__ == "__main__": main()
