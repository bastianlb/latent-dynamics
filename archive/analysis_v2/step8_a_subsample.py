#!/usr/bin/env python
"""Amendment 6 / A: old MNIST set (ii) (69 configs), repeatedly subsample K configs (K=28 -> LOCO training size 27 = confirmatory), ridge LOCO
dR2 for B1 vs B1+G / +compact4 / +NZ18 / +RP18 with the old B1 (B0 + log10_n, log10_wd, label_noise, is_sgd, log10_steps). usage: --K 28 --draws 400 [--part i --nparts n]"""
import step8_common as H
import argparse, csv
import numpy as np
S, M, Z = H.S, H.M, H.Z
ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=28); ap.add_argument("--draws", type=int, default=400)
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
a = ap.parse_args()
old = H.D.load_runs("mnist"); n = len(old)
Pl = np.load(f"{H.C.OUT_ROOT}/ext4/mnist/placebo_feats.npz")
assert Pl["P_100"].shape[0] == n
cfg_all = np.array([r["info"]["config_id"] for r in old]); ucfg = np.unique(cfg_all)
pre = {}
for t in H.TARGETS:
    for cut in H.CUTS:
        y, B0, DES, Gm, g, names = H.K7.get(old, cut, t)
        pre[(t, cut)] = (y, np.hstack([B0, DES]), Gm, M.gvariant_cols(names[2], "compact4"))
rows = []
draws = [d for d in range(a.draws) if d % a.nparts == a.part] if a.K < len(ucfg) else [0]
for d in draws:
    rng = np.random.default_rng(10_000 * a.K + d)
    sel = ucfg if a.K >= len(ucfg) else rng.choice(ucfg, a.K, replace=False)
    mk = np.isin(cfg_all, sel); idx = np.where(mk)[0]
    nz_full = Z.noise_block(n, 18, d)
    for (t, cut), (y, B1, Gm, c4) in pre.items():
        yy, g = y[idx], cfg_all[idx]
        rp = Pl[f"P_{cut}"][idx, d % 5, :]
        sets = {"B1": B1[idx], "G": np.hstack([B1[idx], Gm[idx]]), "c4": np.hstack([B1[idx], Gm[idx][:, c4]]),
                "NZ18": np.hstack([B1[idx], nz_full[idx]]), "RP18": np.hstack([B1[idx], rp])}
        P = {k: S.loco(X, yy, g, "ridge") for k, X in sets.items()}
        r0 = S.r2(yy, P["B1"])
        rows.append(dict(K=a.K, draw=d, target=t, cutoff=cut, n_runs=len(idx), r2_B1=r0, **{f"d_{k}": S.r2(yy, P[k]) - r0 for k in ("G", "c4", "NZ18", "RP18")}))
    if d % 20 == 0: print("K", a.K, "draw", d, flush=True)
fn = f"{H.OUT}/A_subsample_K{a.K}_part{a.part}.csv"
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", fn)
