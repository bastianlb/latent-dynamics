"""Loader/band-metrics for the AE vs classifier linear-dynamical-system comparison (used by scripts/ae/ae_lds_report.py)."""
import os, re, glob
import numpy as np
BV = "/vol/bitbucket/lbastian/latent-dynamics"; VF = f"{BV}/dynamics_v2/vfield"; AEO = f"{BV}/ae_dynamics/analysis/lds"
BANDS = [(1, 10), (10, 100), (100, 1000), (1000, 10 ** 9)]; BN = ["b1_10", "b10_100", "b100_1000", "b1000_up"]
PN = ["R1_affine", "R2_classaffine", "R3_knn", "R3_krr"]


def bmed(x, st, lo, hi):
    k = (st >= lo) & (st < hi); return float(np.nanmedian(x[k])) if k.any() else np.nan


def band_metrics(P, gen, sp, st):
    """P (npart,K) split-mean partition; gen (K,9) or None; sp dict of (nsplit,K) arrays or None; st = steps of interval ends."""
    o = {}
    for bi, ((lo, hi), bn) in enumerate(zip(BANDS, BN)):
        r1, r2, r3 = (bmed(P[i], st, lo, hi) for i in range(3))
        o.update({f"R1_{bn}": r1, f"R2_{bn}": r2, f"R3_{bn}": r3, f"U_{bn}": 1 - r3, f"dR2_{bn}": r2 - r1, f"dR3_{bn}": r3 - r2})
        if gen is not None:
            iso, sh, ro = (gen[:, i] for i in range(3)); tot = iso ** 2 + sh ** 2 + ro ** 2 + 1e-300
            o[f"iso_{bn}"] = bmed(iso, st, lo, hi); o[f"trsh_{bn}"] = bmed(iso ** 2 / tot, st, lo, hi); o[f"symsh_{bn}"] = bmed(sh ** 2 / tot, st, lo, hi); o[f"antish_{bn}"] = bmed(ro ** 2 / tot, st, lo, hi)
            o[f"Lnorm_{bn}"] = bmed(np.sqrt(tot), st, lo, hi); o[f"imag_{bn}"] = bmed(gen[:, 7], st, lo, hi); o[f"emax_{bn}"] = bmed(gen[:, 3], st, lo, hi)
        if sp is not None:
            for k in ("glob_abscissa", "glob_n_exp", "glob_n_cplx", "abscissa_med_tot", "n_exp_tot", "n_cplx_tot", "logcond_tot", "abscissa_med_res", "logcond_res"):
                if k in sp: o[f"{k}_{bn}"] = bmed(np.nanmean(sp[k], 0), st, lo, hi)
    return o


def clf_gen(rid, probe):
    """global generator rows for a classifier run (computed here when the stored decomp lacks them; cached)."""
    from latentdyn.vfield import core as V, decomp as VD
    c = f"{AEO}/clf_gen_cache/{rid}_{probe}.npy"
    if os.path.isfile(c): return np.load(c)
    Z, steps, _, _ = V.load_run(f"{V.RUNS}/{rid}", probe); G, _ = VD.generator(Z, steps); os.makedirs(os.path.dirname(c), exist_ok=True); np.save(c, G); return G


def load_classifier(dataset, probe="test"):
    """classifier phase runs (test probe), dataset in {mnist, fashion_mnist}; returns list of dict(N, wd, seed, **band metrics)."""
    base = f"{VF}/decomp" if dataset == "mnist" else f"{VF}/fm/partition"; sbase = f"{VF}/spectra" if dataset == "mnist" else f"{VF}/fm/spectra"; rows = []
    for f in sorted(glob.glob(f"{base}/runs/{dataset}_probe_none_d16_s*_phase_n*_wd*_{probe}.npz")):
        m = re.search(r"_s(\d+)_phase_n(\w+?)_wd([\d.e-]+)_(?:test|train)\.npz$", f); seed, N, wd = m.groups(); z = np.load(f, allow_pickle=True)
        rid = os.path.basename(f)[: -len(probe) - 5]; gen = z["gen"] if "gen" in z.files else clf_gen(rid, probe)
        spf = f"{sbase}/runs/{os.path.basename(f)[:-4]}_given.npz"; sp = None
        if os.path.isfile(spf):
            zs = np.load(spf); sp = {k: zs[k] for k in zs.files if zs[k].ndim == 2 and zs[k].shape[1] == len(z["steps"]) - 1}
        st = z["steps"][1:]; r = dict(kind="clf", dataset=dataset, probe=probe, N=(60000 if N == "full" else int(N)), wd=float(wd), seed=int(seed), arch="clf"); r.update(band_metrics(z["part_mean"], gen, sp, st)); r["steps"] = z["steps"]
        r["_P"] = z["part_mean"][:3]; r["_gen"] = gen; rows.append(r)
    return rows


def load_ae(mode="lab"):
    rows = []
    for f in sorted(glob.glob(f"{AEO}/runs/*__{mode}.npz")):
        rid, probe, _ = os.path.basename(f)[:-4].split("__"); m = re.match(r"(mnist|fashion_mnist)_ae_(mlp|linear)_d(\d+)_s(\d+)_n(\w+?)_wd([\d.e-]+)$", rid)
        if not m: continue
        ds, arch, d, seed, N, wd = m.groups(); z = np.load(f); st = z["steps"][1:]
        sp = {k[3:]: z[k] for k in z.files if k.startswith("sp_") and z[k].ndim == 2}
        r = dict(kind="ae", dataset=ds, probe=probe, arch=arch, d=int(d), N=(60000 if N == "full" else int(N)), wd=float(wd), seed=int(seed), rid=rid)
        r.update(band_metrics(z["part_mean"], z["gen"] if "gen" in z.files else None, sp, st)); r["steps"] = z["steps"]; r["_P"] = z["part_mean"][:3]
        if "gen" in z.files: r["_gen"] = z["gen"]
        rc = {k: z["rc_" + k] for k in ("recon_train", "recon_test", "recon_probe")}; r["rc"] = rc; r["gap_final"] = float(rc["recon_test"][-1] - rc["recon_train"][-1]); r["rtest_final"] = float(rc["recon_test"][-1]); r["rtrain_final"] = float(rc["recon_train"][-1])
        rows.append(r)
    return rows
