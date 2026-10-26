#!/usr/bin/env python
"""Amendment 6 (post hoc) B: harm p-values (one-sided p for dR2<0, Holm over 10 cells, families full G and compact4) next to the pre-registered p+ Holm values;
C: bootstrap of the mean (G - placebo) contrast, draw-averaging inside each bootstrap rep (RP18 and NZ18, 5 draws each), Holm over 10 cells;
D: confirmatory precision (90% CI half-width per cell) and config-resampling simulation at sizes 28/56/112(/224/448)."""
import step8_common as H
import csv, os
import numpy as np
S, M, Z, K7 = H.S, H.M, H.Z, H.K7
D = H.D
holm = K7.holm
new = H.load_new(); n = len(new); WS = Z.make_dirs(); nd = Z.NDRAW
fp = f"{H.OUT}/conf_rp_feats.npz"
if os.path.exists(fp):
    z = np.load(fp); Pf = {c: z[f"P_{c}"] for c in H.CUTS}
else:
    Pf = {}
    for cut in H.CUTS:
        P = np.zeros((n, nd, 18))
        for i, r in enumerate(new):
            st_ = r["step"]; k = D.idx_le_mask(st_, r["topo"], cut); rid = r["info"]["run_id"]; f = {}
            for ps in ("test", "train"):
                x = np.load(Z.snap_files(rid, ps, st_[k]))["latents"].astype(np.float64)
                f[ps] = np.stack([Z.proj_stats(x, W) for W in WS])
            P[i] = np.hstack([f["test"], f["train"], f["train"] - f["test"]])
        Pf[cut] = P
    np.savez(fp, **{f"P_{c}": Pf[c] for c in H.CUTS})
print("features ready", flush=True)


def write(fn, rows):
    with open(f"{H.OUT}/{fn}", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)


ref = {(r["target"], int(r["cutoff"]), r["X"]): r for r in csv.DictReader(open(f"{H.C.OUT_ROOT}/confirm/results/primary_ridge.csv"))}
rowsB, rowsC, rowsD, rowsDsim = [], [], [], []
for t in H.TARGETS:
    m = H.margin(t)
    for cut in H.CUTS:
        y, B1, G, names, g, B0 = H.cut_data(new, cut, t)
        c4 = M.gvariant_cols(names[2], "compact4")
        P = {"b": S.loco(B1, y, g, "ridge"), "F": S.loco(np.hstack([B1, G]), y, g, "ridge"), "C": S.loco(np.hstack([B1, G[:, c4]]), y, g, "ridge")}
        for d_ in range(nd):
            P[f"rp{d_}"] = S.loco(np.hstack([B1, Pf[cut][:, d_, :]]), y, g, "ridge")
            P[f"nz{d_}"] = S.loco(np.hstack([B1, Z.noise_block(n, 18, d_)]), y, g, "ridge")
        st = M.CfgStats(y, P, g)
        W = st.draw(H.NB, np.random.default_rng(0)); W1 = np.ones((1, st.nc))
        # ---- B
        for k, X in (("F", "G"), ("C", "compact4")):
            d = st.d_r2("b", k, W); obs = st.d_r2("b", k, W1)[0]
            r0 = ref[(t, cut, X)]
            assert abs(obs - float(r0["delta"])) < 1e-9 and abs((np.sum(d <= 0) + 1) / (H.NB + 1) - float(r0["p_pos"])) < 1e-12, "does not reproduce step7"
            q = np.percentile(d, [5, 95])
            rowsB.append(dict(X=X, target=t, cutoff=cut, delta=obs, lo90=q[0], hi90=q[1], p_neg=(np.sum(d >= 0) + 1) / (H.NB + 1),
                              p_pos_prereg=float(r0["p_pos"]), pos_holm_prereg=float(r0["pos_holm"]), tost_holm_prereg=float(r0["tost_holm"])))
            hw = (q[1] - q[0]) / 2
            rowsD.append(dict(X=X, target=t, cutoff=cut, margin=m, delta=obs, lo90=q[0], hi90=q[1], halfwidth90=hw,
                              K_needed_for_hw_below_margin_invsqrt=28 * (hw / m) ** 2))
        # ---- C
        for kind, pre in (("RP18", "rp"), ("NZ18", "nz")):
            cs = np.mean([st.d_r2(f"{pre}{d_}", "F", W) for d_ in range(nd)], 0)        # R2(G)-R2(placebo), draw-mean inside each rep
            c_obs = np.mean([st.d_r2(f"{pre}{d_}", "F", W1)[0] for d_ in range(nd)])
            ps = np.mean([st.d_r2("b", f"{pre}{d_}", W) for d_ in range(nd)], 0); ps_obs = np.mean([st.d_r2("b", f"{pre}{d_}", W1)[0] for d_ in range(nd)])
            q = np.percentile(cs, [2.5, 97.5, 5, 95]); qp = np.percentile(ps, [2.5, 97.5])
            rowsC.append(dict(placebo=kind, target=t, cutoff=cut, G_minus_placebo=c_obs, lo95=q[0], hi95=q[1], lo90=q[2], hi90=q[3],
                              p_G_worse=(np.sum(cs >= 0) + 1) / (H.NB + 1), p_G_better=(np.sum(cs <= 0) + 1) / (H.NB + 1),
                              placebo_dR2_vs_B1=ps_obs, placebo_dR2_lo95=qp[0], placebo_dR2_hi95=qp[1],
                              placebo_p_neg=(np.sum(ps >= 0) + 1) / (H.NB + 1)))
        # ---- D simulation (plug-in config resampling at larger sizes; OOF predictions held fixed)
        rng = np.random.default_rng(1)
        for k, X in (("F", "G"), ("C", "compact4")):
            for K in (28, 56, 112, 224, 448):
                hws = []; ds = []
                for _ in range(150):
                    wo = np.bincount(rng.integers(0, st.nc, K), minlength=st.nc).astype(float)       # outer: K configs with replacement
                    wi = rng.multinomial(K, wo / K, size=600).astype(float)                          # inner bootstrap of that K-config sample
                    d = st.d_r2("b", k, wi); q = np.percentile(d, [5, 95]); hws.append((q[1] - q[0]) / 2)
                    ds.append(st.d_r2("b", k, wo[None])[0])
                rowsDsim.append(dict(X=X, target=t, cutoff=cut, margin=m, K=K, hw90_mean=float(np.mean(hws)), hw90_sd=float(np.std(hws)), delta_mean=float(np.mean(ds))))
        print(t, cut, flush=True)
for X in ("G", "compact4"):
    fam = [r for r in rowsB if r["X"] == X]
    for r, h in zip(fam, holm([r["p_neg"] for r in fam])): r["p_neg_holm"] = h
for kind in ("RP18", "NZ18"):
    fam = [r for r in rowsC if r["placebo"] == kind]
    for r, h in zip(fam, holm([r["p_G_worse"] for r in fam])): r["p_G_worse_holm"] = h
    for r, h in zip(fam, holm([r["placebo_p_neg"] for r in fam])): r["placebo_p_neg_holm"] = h
write("B_harm_pvalues.csv", rowsB); write("C_placebo_contrast.csv", rowsC); write("D_precision.csv", rowsD); write("D_sim_halfwidth.csv", rowsDsim)
print("done")
