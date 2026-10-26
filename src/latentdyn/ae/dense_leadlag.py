"""Lead/lag analysis of latent-geometry signatures vs the generalization gap, on the DENSE early-window AE runs.  CPU only, <=8 threads.

Usage: python ae_dense_leadlag.py [--stage compute|report|all] [--threads 8]
Input : ae_dynamics/dense/runs/<run>/dense_ae.npz (every 2 steps; latents f32, per-sample recon_err, recon_train/test) + grid runs for verification
Output: ae_dynamics/dense/analysis/{series/<run>.npz, results.json, verify.json, fig_*.png}

Signatures (computed at EVERY dense snapshot; train probe = the N pool points (250 or 1000), test probe = 1000 test points):
  LATENT-ONLY (no decoder / no recon error involved):
    spread_lab  pooled within-class variance, train / test   (test subsampled label-stratified to n_tr when n_tr<1000)
    var_ratio   total variance train / test
    effrank     participation ratio of centred singular values^2, train / test
    nn_ratio    median_test min_train ||z-z'|| / median_train min_other-train ||z-z'||
    vel_ratio   median ||z(t+w)-z(t-w)|| train / test (both normalised by test latent scale)
    aff_diff    (1 - adjR2 of a GLOBAL affine fit v = A z + b over the points) train minus test (residual share of the velocity field)
  DECODER-DEPENDENT (tied to the recon gap by construction, reported but flagged):
    dec_lab / dec_km  Spearman(err_i, LOO distance to class / input-cluster mean) test minus train
  Pixel-PCA reference: the same ratios for train-pool PCA-16 projections of the same images (constant in time).
Targets: gap = recon_test - recon_train (abs), gap_rel = te/tr-1, test recon and its minimum t* (smoothed).
Onset of a signature s: baseline b = median(s) over epochs [2,6]; L = s(t*); progress p=(s-b)/(L-b);
  t_f = (last time before t* with p<f) -> next sample ("final crossing", robust to early noise), f in {0.25, 0.5}.
  Signatures with |L-b| < 5*sd(baseline window) are flagged undetectable.  Epoch = step*128/N.
Lead/lag: Gaussian-smoothed signals on a log-time grid, cross-correlation of d s/d ln t with d gap/d ln t over lags in ln t (lag>0 => signature LAGS the gap).
Bootstrap: resample the 3 seeds (paired) 4000x; CI = 2.5/97.5 percentiles of the seed-mean (very coarse with n=3, per-seed values also listed).
"""
import os
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, json
import numpy as np
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter1d, median_filter
from latentdyn.ae import analysis as A
from latentdyn.stats import spearman_fast, aff_resid, boot_ci  # noqa: F401  (re-exported; ae_dense_null imports them)

ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics"
DENSE = f"{ROOT}/dense"; GRID = f"{ROOT}/runs"; OUT = f"{DENSE}/analysis"
CONFIGS = [(1000, "0"), (1000, "1e-4"), (250, "0"), (250, "1e-4")]
SEEDS = (0, 1, 2)
W = 3  # velocity half-window in snapshots (3 snapshots = 6 steps each side)
SIGS = ["spread_lab", "var_ratio", "effrank", "nn_ratio", "vel_ratio", "aff_diff", "dec_lab", "dec_km"]
LATENT_ONLY = SIGS[:6]


def rid(N, wd, s): return f"mnist_ae_mlp_d16_s{s}_n{N}_wd{wd}"


# ------------------------------------------------------------------ per-snapshot statistics
def pooled_within(Z, g):
    ss = 0.0; k = 0; n = 0
    for c in np.unique(g):
        zc = Z[g == c]
        if len(zc) < 2: continue
        ss += ((zc - zc.mean(0)) ** 2).sum(); k += 1; n += len(zc)
    return ss / max(n - k, 1)


def totvar(Z): return ((Z - Z.mean(0)) ** 2).sum() / (len(Z) - 1)


def effrank1(Z):
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2; return s.sum() ** 2 / (s ** 2).sum()


def nn_med(X, Y, same=False):
    D = A.sqd(X, Y)
    if same: np.fill_diagonal(D, np.inf)
    return np.sqrt(np.median(D.min(1)))


def loo_dist(Z, g):
    """LOO distance of each point to its group mean (vectorised)."""
    d = np.full(len(Z), np.nan)
    for c in np.unique(g):
        ii = np.where(g == c)[0]; n = len(ii)
        if n < 3: continue
        s = Z[ii].sum(0)
        d[ii] = np.linalg.norm(Z[ii] - (s - Z[ii]) / (n - 1), axis=1)
    return d






def pixel_pca_ref(rd, ntr, subs, ytr, yte, Xtr_pool, Xtr, Xte):
    """same ratios for PCA-16 (fit on the train pool) projections of the probe images."""
    mu = Xtr_pool.mean(0); U, S, Vt = np.linalg.svd(Xtr_pool - mu, full_matrices=False); P = Vt[:16]
    zt, ze = (Xtr - mu) @ P.T, (Xte - mu) @ P.T
    return dict(spread_lab=np.mean([pooled_within(zt, ytr) / pooled_within(ze[ix], yte[ix]) for ix in subs]),
                var_ratio=np.mean([totvar(zt) / totvar(ze[ix]) for ix in subs]),
                effrank=np.mean([effrank1(zt) / effrank1(ze[ix]) for ix in subs]),
                nn_ratio=nn_med(ze, zt) / nn_med(zt, zt, True))


def compute_run(N, wd, seed):
    rd = f"{DENSE}/runs/{rid(N, wd, seed)}"; Dd = np.load(f"{rd}/dense_ae.npz")
    steps = Dd["step"]; Zte, Ztr = Dd["z_test"].astype(np.float64), Dd["z_train"].astype(np.float64)
    Ete, Etr = Dd["err_test"].astype(np.float64), Dd["err_train"].astype(np.float64)
    yte, ytr = Dd["labels_test"], Dd["labels_train"]; T, nte, d = Zte.shape; ntr = Ztr.shape[1]
    rs = np.random.RandomState(0)
    Xte_all, _ = A.images("mnist", False); Xtr_all, _ = A.images("mnist", True)
    pte = np.load(f"{rd}/probe_index.npz")["indices"]; ptr = np.load(f"{rd}/probe_train_index.npz")["indices"]
    pool = np.load(f"{rd}/train_index.npz")["indices"]
    Xte, Xtr = Xte_all[pte].astype(np.float64), Xtr_all[ptr].astype(np.float64)
    assign = A.group_model("mnist", f"{ROOT}/analysis/_cache"); gte_km, gtr_km = assign(Xte), assign(Xtr)
    subs = []
    if ntr < nte:
        for _ in range(3):
            idx = []
            for c in range(10):
                ic = np.where(yte == c)[0]; k = int(round(ntr * (ytr == c).mean())); idx += list(rs.choice(ic, min(k, len(ic)), replace=False))
            subs.append(np.array(sorted(idx)))
    else:
        subs = [np.arange(nte)]
    ref = pixel_pca_ref(rd, ntr, subs, ytr, yte, Xtr_all[pool].astype(np.float64), Xtr, Xte)
    R = {k: np.full(T, np.nan) for k in SIGS}
    for t in range(T):
        zt, ze = Ztr[t], Zte[t]
        R["spread_lab"][t] = np.mean([pooled_within(zt, ytr) / pooled_within(ze[ix], yte[ix]) for ix in subs])
        R["var_ratio"][t] = np.mean([totvar(zt) / totvar(ze[ix]) for ix in subs])
        R["effrank"][t] = np.mean([effrank1(zt) / effrank1(ze[ix]) for ix in subs])
        R["nn_ratio"][t] = nn_med(ze, zt) / (nn_med(zt, zt, True) + 1e-300)
        R["dec_lab"][t] = spearman_fast(Ete[t], loo_dist(ze, yte)) - spearman_fast(Etr[t], loo_dist(zt, ytr))
        R["dec_km"][t] = spearman_fast(Ete[t], loo_dist(ze, gte_km)) - spearman_fast(Etr[t], loo_dist(zt, gtr_km))
        if W <= t < T - W:
            vt, ve = Ztr[t + W] - Ztr[t - W], Zte[t + W] - Zte[t - W]
            sc = np.sqrt(totvar(ze))
            R["vel_ratio"][t] = np.median(np.linalg.norm(vt, axis=1)) / np.median(np.linalg.norm(ve, axis=1))
            R["aff_diff"][t] = aff_resid(zt, vt) - aff_resid(ze, ve)
    out = dict(steps=steps, N=N, wd=wd, seed=seed, recon_train=Dd["recon_train"], recon_test=Dd["recon_test"],
               probe_te=Ete.mean(1), probe_tr=Etr.mean(1), **R, **{f"ref_{k}": v for k, v in ref.items()})
    np.savez(f"{OUT}/series/{rid(N, wd, seed)}.npz", **out)
    return out


# ------------------------------------------------------------------ verification against the grid
def verify():
    res = {}
    for N, wd in CONFIGS:
        for s in SEEDS:
            r = rid(N, wd, s); g, dd = f"{GRID}/{r}", f"{DENSE}/runs/{r}"
            row = {}
            # (a) log-schedule snapshots of the dense run vs the grid run (identical step files)
            dmax_z, dmax_s, nstep = 0.0, 0.0, 0
            for f in sorted(glob.glob(f"{g}/latents_train/step_*.npz")):
                h = f"{dd}/latents_train/{os.path.basename(f)}"
                if not os.path.isfile(h): continue
                a, b = np.load(f), np.load(h); nstep += 1
                dmax_z = max(dmax_z, float(np.abs(a["latents"] - b["latents"]).max()))
                for k in ("recon_train", "recon_test", "recon_probe_train", "recon_probe_test"):
                    dmax_s = max(dmax_s, abs(float(a[k]) - float(b[k])))
            row.update(n_shared_log_steps=nstep, max_abs_latent_diff=dmax_z, max_abs_recon_diff=dmax_s)
            # (b) dense_ae.npz values at steps that exist in the grid (log schedule)
            Dd = np.load(f"{dd}/dense_ae.npz"); st = list(Dd["step"]); dz, de = 0.0, 0.0; n2 = 0
            for f in sorted(glob.glob(f"{g}/latents_train/step_*.npz")):
                a = np.load(f); sp = int(a["step"])
                if sp in st:
                    i = st.index(sp); n2 += 1
                    dz = max(dz, float(np.abs(a["latents"] - Dd["z_train"][i]).max()))
                    de = max(de, abs(float(a["recon_train"]) - float(Dd["recon_train"][i])), abs(float(a["recon_test"]) - float(Dd["recon_test"][i])))
            row.update(n_dense_vs_grid_steps=n2, dense_vs_grid_latent_maxdiff=dz, dense_vs_grid_recon_maxdiff=de)
            res[r] = row
    json.dump(res, open(f"{OUT}/verify.json", "w"), indent=1)
    return res


# ------------------------------------------------------------------ onset / lead-lag
def smooth_ts(x, k=5):
    ok = np.isfinite(x)
    if ok.sum() < 3: return x
    y = x.copy(); y[ok] = median_filter(x[ok], size=k, mode="nearest"); return y


def tstar(S, N):
    """step of the minimum of the smoothed test recon."""
    sm = median_filter(S["recon_test"], size=9, mode="nearest"); i = int(np.argmin(sm)); return i, sm


def onset_times(S, N, sig_vals, istar, fracs=(0.25, 0.5)):
    steps = S["steps"].astype(float); ep = steps * 128 / N
    s = smooth_ts(sig_vals); base = (ep >= 2) & (ep <= 6) & np.isfinite(s)
    b, sd = np.median(s[base]), np.std(sig_vals[base]) + 1e-12; L = s[istar]
    out = dict(b=float(b), L=float(L), snr=float(abs(L - b) / sd))
    if not np.isfinite(L) or abs(L - b) < 5 * sd:
        for f in fracs: out[f"t{int(f*100)}"] = np.nan
        out["detectable"] = False; return out
    p = (s - b) / (L - b); out["detectable"] = True
    for f in fracs:
        below = np.where((p[:istar + 1] < f) & np.isfinite(p[:istar + 1]))[0]
        j = below[-1] + 1 if len(below) else 0
        out[f"t{int(f*100)}"] = float(ep[min(j, istar)])  # epoch
    return out


def hinge_onset(ep, x, istar, lo=5.0, n=120):
    """broken-stick fit in ln t on [lo, t*]: y = c + a*max(0, ln t - ln tau). returns (tau epoch, R2 vs constant, rise/resid_sd, detectable)."""
    sm = smooth_ts(x); ok = np.isfinite(sm)
    g = np.exp(np.linspace(np.log(lo), np.log(ep[istar]), n)); y = np.interp(np.log(g), np.log(ep[ok] + 1e-9), sm[ok]); lg = np.log(g)
    sst = ((y - y.mean()) ** 2).sum() + 1e-300; best = None
    for j in range(3, n - 8):
        h = np.maximum(0, lg - lg[j]); X = np.stack([np.ones(n), h], 1); B, *_ = np.linalg.lstsq(X, y, rcond=None); sse = ((y - X @ B) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, g[j], B[1], B[1] * (lg[-1] - lg[j]))
    sse, tau, a, rise = best; rsd = np.sqrt(sse / n) + 1e-300; r2 = 1 - sse / sst
    return float(tau), float(r2), float(rise / rsd), bool(r2 > 0.5 and abs(rise) > 3 * rsd), float(np.sign(rise))


def gap_signals(S):
    gap = S["recon_test"] - S["recon_train"]; rel = S["recon_test"] / np.maximum(S["recon_train"], 1e-9) - 1
    return gap, rel


def first_cross(ep, x, thr, istar=None):
    """first epoch where smoothed x stays >= thr (final-crossing definition up to end of window)."""
    s = smooth_ts(x); below = np.where(s < thr)[0]; j = below[-1] + 1 if len(below) else 0
    return float(ep[min(j, len(ep) - 1)])


def logt_grid(ep, lo=3.0, n=300):
    return np.exp(np.linspace(np.log(lo), np.log(ep[-1]), n))


def xcorr_lag(ep, a, b, lo=3.0, maxlag=0.8, sig=0.04):
    """lag (ln t) of a relative to b via cross-correlation of d/dlnt of smoothed series. lag>0: a lags b."""
    g = logt_grid(ep, lo); ok = np.isfinite(a) & np.isfinite(b)
    ia = np.interp(np.log(g), np.log(ep[ok] + 1e-9), a[ok]); ib = np.interp(np.log(g), np.log(ep[ok] + 1e-9), b[ok])
    dl = np.log(g[1] / g[0]); da = np.gradient(gaussian_filter1d(ia, sig / dl), dl); db = np.gradient(gaussian_filter1d(ib, sig / dl), dl)
    da = (da - da.mean()) / (da.std() + 1e-300); db = (db - db.mean()) / (db.std() + 1e-300)
    K = int(maxlag / dl); lags = np.arange(-K, K + 1); cc = []
    for k in lags:
        x, y = (da[k:], db[:len(db) - k]) if k >= 0 else (da[:k], db[-k:])
        cc.append(np.mean(x * y))
    cc = np.array(cc); i = int(np.argmax(cc))
    if 0 < i < len(cc) - 1:
        den = cc[i - 1] - 2 * cc[i] + cc[i + 1]; off = 0.5 * (cc[i - 1] - cc[i + 1]) / den if den != 0 else 0.0
    else: off = 0.0
    return float((lags[i] + off) * dl), float(cc[i])




def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = dict(); fig_data = {}
    for N, wd in CONFIGS:
        key = f"N{N}_wd{wd}"; per = []
        for s in SEEDS:
            S = dict(np.load(f"{OUT}/series/{rid(N, wd, s)}.npz")); ep = S["steps"] * 128 / N
            i, smt = tstar(S, N); gap, rel = gap_signals(S); row = dict(seed=s, tstar_ep=float(ep[i]), tstar_step=int(S["steps"][i]), test_min=float(smt[i]),
                                                                       test_final_window=float(S["recon_test"][-1]), gap_at_tstar=float(gap[i]))
            row["gap"] = onset_times(S, N, gap, i); row["gap_rel_10pct_ep"] = first_cross(ep, rel, 0.10)
            # test recon within 5% / 2% of its minimum (stall indicator)
            row["test_5pct_ep"] = float(ep[np.where(median_filter(S["recon_test"], size=9, mode="nearest") <= 1.05 * smt[i])[0][0]])
            row["test_2pct_ep"] = float(ep[np.where(median_filter(S["recon_test"], size=9, mode="nearest") <= 1.02 * smt[i])[0][0]])
            row["sig"] = {}
            hg = hinge_onset(ep, gap, i); row["gap_hinge"] = dict(tau=hg[0], r2=hg[1], snr=hg[2], det=hg[3])
            for k in SIGS:
                o = onset_times(S, N, S[k], i); o["val_tstar"] = float(smooth_ts(S[k])[i]); o["ref"] = float(S.get(f"ref_{k}", np.nan)) if f"ref_{k}" in S else None
                if o["detectable"]:
                    o["lag_t50_minus_gap_t50_ep"] = o["t50"] - row["gap"]["t50"] if row["gap"]["detectable"] else np.nan
                    o["lag_t50_minus_tstar_ep"] = o["t50"] - row["tstar_ep"]
                    o["ln_ratio_t50_gap"] = float(np.log(o["t50"] / row["gap"]["t50"])) if row["gap"]["detectable"] else np.nan
                    o["ln_ratio_t25_gap25"] = float(np.log(o["t25"] / row["gap"]["t25"])) if row["gap"]["detectable"] else np.nan
                hs = hinge_onset(ep, S[k], i); o.update(hinge_tau=hs[0], hinge_r2=hs[1], hinge_snr=hs[2], hinge_det=hs[3], hinge_sign=hs[4],
                                                         hinge_lnratio_gap=float(np.log(hs[0] / hg[0])) if hs[3] else np.nan,
                                                         hinge_lnratio_tstar=float(np.log(hs[0] / row["tstar_ep"])) if hs[3] else np.nan)
                lo = 3.0
                o["xcorr_lag_vs_gap_lnt"], o["xcorr_peak_vs_gap"] = xcorr_lag(ep, smooth_ts(S[k]), smooth_ts(gap), lo)
                o["xcorr_lag_vs_test_lnt"], o["xcorr_peak_vs_test"] = xcorr_lag(ep, smooth_ts(S[k]), smooth_ts(S["recon_test"]), lo)
                row["sig"][k] = o
            row["gap_xcorr_vs_test_lnt"], _ = xcorr_lag(ep, smooth_ts(gap), smooth_ts(S["recon_test"]), 3.0)
            per.append(row); fig_data[(N, wd, s)] = (S, i)
        agg = dict(gap_hinge=boot_ci([r["gap_hinge"]["tau"] for r in per]), gap_hinge_r2=boot_ci([r["gap_hinge"]["r2"] for r in per]), tstar_ep=boot_ci([r["tstar_ep"] for r in per]), gap_t25=boot_ci([r["gap"]["t25"] for r in per]), gap_t50=boot_ci([r["gap"]["t50"] for r in per]),
                   gap_rel10=boot_ci([r["gap_rel_10pct_ep"] for r in per]), test_5pct=boot_ci([r["test_5pct_ep"] for r in per]), test_2pct=boot_ci([r["test_2pct_ep"] for r in per]),
                   test_min=boot_ci([r["test_min"] for r in per]), gap_at_tstar=boot_ci([r["gap_at_tstar"] for r in per]), sig={})
        for k in SIGS:
            g = lambda f: [r["sig"][k].get(f, np.nan) for r in per]
            agg["sig"][k] = dict(n_detectable=int(sum(r["sig"][k]["detectable"] for r in per)), t25=boot_ci(g("t25")), t50=boot_ci(g("t50")),
                                 lag_t50_vs_gap_ep=boot_ci(g("lag_t50_minus_gap_t50_ep")), lag_t50_vs_tstar_ep=boot_ci(g("lag_t50_minus_tstar_ep")),
                                 lnratio_t50_vs_gap=boot_ci(g("ln_ratio_t50_gap")), lnratio_t25_vs_gap25=boot_ci(g("ln_ratio_t25_gap25")),
                                 xcorr_lag_vs_gap_lnt=boot_ci(g("xcorr_lag_vs_gap_lnt")), xcorr_peak_vs_gap=boot_ci(g("xcorr_peak_vs_gap")),
                                 xcorr_lag_vs_test_lnt=boot_ci(g("xcorr_lag_vs_test_lnt")),
                                 hinge_tau=boot_ci(g("hinge_tau")), hinge_r2=boot_ci(g("hinge_r2")), hinge_n_det=int(sum(bool(x) for x in g("hinge_det"))),
                                 hinge_lnratio_gap=boot_ci(g("hinge_lnratio_gap")), hinge_lnratio_tstar=boot_ci(g("hinge_lnratio_tstar")),
                                 val_tstar=boot_ci(g("val_tstar")), snr=boot_ci(g("snr")), ref=per[0]["sig"][k]["ref"])
        agg["gap_xcorr_vs_test_lnt"] = boot_ci([r["gap_xcorr_vs_test_lnt"] for r in per])
        res[key] = dict(per_seed=per, agg=agg)
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=lambda o: None if o is None else float(o))
    figures(fig_data, res); write_md(res)


def figures(fig_data, res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = dict(spread_lab="C0", var_ratio="C1", effrank="C2", nn_ratio="C3", vel_ratio="C4", aff_diff="C5", dec_lab="C6", dec_km="C7")
    # fig 1: raw time series
    fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharex="col")
    for j, (N, wd) in enumerate(CONFIGS):
        for s in SEEDS:
            S, i = fig_data[(N, wd, s)]; ep = S["steps"] * 128 / N; gap, _ = gap_signals(S)
            ax[0, j].plot(ep, S["recon_test"], "C3", alpha=.7, lw=.9, label="test" if s == 0 else None)
            ax[0, j].plot(ep, S["recon_train"], "C0", alpha=.7, lw=.9, label="train" if s == 0 else None)
            ax[0, j].plot(ep, gap, "k", alpha=.6, lw=.9, label="gap" if s == 0 else None); ax[0, j].axvline(ep[i], color="gray", ls=":", lw=.7)
            for k in ("spread_lab", "nn_ratio", "effrank", "var_ratio"):
                ax[1, j].plot(ep, smooth_ts(S[k]), cols[k], alpha=.7, lw=.9, label=k if s == 0 else None)
        ax[0, j].set_xscale("log"); ax[0, j].set_title(f"N={N} wd={wd} (dotted: test-min t*)"); ax[0, j].set_yscale("log"); ax[1, j].set_xscale("log"); ax[1, j].set_xlabel("epoch")
        ax[1, j].axhline(1, color="gray", lw=.5)
    ax[0, 0].legend(); ax[1, 0].legend(fontsize=7); ax[0, 0].set_ylabel("recon MSE"); ax[1, 0].set_ylabel("train/test ratio")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_timeseries.png", dpi=110); plt.close(fig)
    # fig 2: normalised progress p(t) of all signatures vs gap, mean over seeds
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for j, (N, wd) in enumerate(CONFIGS):
        for k in ["gap"] + SIGS:
            ps = []
            for s in SEEDS:
                S, i = fig_data[(N, wd, s)]; ep = S["steps"] * 128 / N
                x = gap_signals(S)[0] if k == "gap" else S[k]; sm = smooth_ts(x); base = (ep >= 2) & (ep <= 6) & np.isfinite(sm)
                b = np.median(sm[base]); L = sm[i]; p = (sm - b) / (L - b) if np.isfinite(L) and abs(L - b) > 1e-12 else np.full_like(sm, np.nan)
                g = np.exp(np.linspace(np.log(3), np.log(ep[i] * 2.5), 200)); ps.append(np.interp(np.log(g), np.log(ep + 1e-9), p))
            ax[j].plot(g, np.nanmean(ps, 0), "k" if k == "gap" else cols[k], lw=2.6 if k == "gap" else 1.1, ls="-" if (k in LATENT_ONLY or k == "gap") else "--", label=k)
        ax[j].axhline(.5, color="gray", lw=.5); ax[j].axvline(np.mean([fig_data[(N, wd, s)][0]["steps"][fig_data[(N, wd, s)][1]] * 128 / N for s in SEEDS]), color="gray", ls=":")
        ax[j].set_xscale("log"); ax[j].set_ylim(-.5, 1.6); ax[j].set_title(f"N={N} wd={wd}: progress p(t) (1 = value at t*)"); ax[j].set_xlabel("epoch")
    ax[0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(f"{OUT}/fig_progress.png", dpi=110); plt.close(fig)
    # fig 3: onset t50 per signature, per config (seed means + CI), relative to gap t50 and t*
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3), sharey=True)
    for j, (N, wd) in enumerate(CONFIGS):
        a = res[f"N{N}_wd{wd}"]["agg"]; ys = ["gap"] + SIGS
        for y, k in enumerate(ys):
            m = a["gap_t50"] if k == "gap" else a["sig"][k]["t50"]
            if np.isfinite(m[0]): ax[j].errorbar(m[0], y, xerr=[[m[0] - m[1]], [m[2] - m[0]]], fmt="o", color="k" if k == "gap" else cols[k])
            for r in res[f"N{N}_wd{wd}"]["per_seed"]:
                v = r["gap"]["t50"] if k == "gap" else r["sig"][k].get("t50", np.nan)
                if v is not None and np.isfinite(v): ax[j].plot(v, y + .2, ".", color="gray", ms=4)
        ax[j].axvline(a["tstar_ep"][0], color="r", ls=":", label="test min t*"); ax[j].set_xscale("log"); ax[j].set_yticks(range(len(ys))); ax[j].set_yticklabels(ys)
        ax[j].set_title(f"N={N} wd={wd}: t50 (epoch)"); ax[j].set_xlabel("epoch")
    ax[0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig_onsets.png", dpi=110); plt.close(fig)
    # fig 4: xcorr lags
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3), sharey=True)
    for j, (N, wd) in enumerate(CONFIGS):
        a = res[f"N{N}_wd{wd}"]["agg"]
        for y, k in enumerate(SIGS):
            m = a["sig"][k]["xcorr_lag_vs_gap_lnt"]
            if np.isfinite(m[0]): ax[j].errorbar(m[0], y, xerr=[[m[0] - m[1]], [m[2] - m[0]]], fmt="o", color=cols[k])
        ax[j].axvline(0, color="k", lw=.7); ax[j].set_yticks(range(len(SIGS))); ax[j].set_yticklabels(SIGS); ax[j].set_title(f"N={N} wd={wd}: xcorr lag vs gap (ln t; >0 lags)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_xcorr.png", dpi=110); plt.close(fig)


def f(x, p=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def ci(c, p=1): return f"{f(c[0], p)} [{f(c[1], p)}, {f(c[2], p)}]" if c[3] else "n/a"


def write_md(res):
    L = ["# Dense early-window AE runs: does latent geometry lead or lag generalization?", "",
         "MNIST AE d=16, same hyperparameters/seeds as the grid, snapshots every 2 steps (N=1000 to step 4000 = epoch 512; N=250 to step 2000 = epoch 1000). "
         "Epoch = step*128/N. 3 seeds per config; entries are seed-mean [bootstrap 95% CI over seeds] (n=3: coarse, treat as a spread, not an inferential CI). Definitions: see docstring of ae_dense_leadlag.py.", ""]
    ver = json.load(open(f"{OUT}/verify.json")); mz = max(v["max_abs_latent_diff"] for v in ver.values()); mr = max(v["max_abs_recon_diff"] for v in ver.values())
    dz = max(v["dense_vs_grid_latent_maxdiff"] for v in ver.values()); dr = max(v["dense_vs_grid_recon_maxdiff"] for v in ver.values())
    L += ["## Verification vs grid", f"All 12 dense runs: log-schedule snapshots vs grid files: max |latent diff| = {mz:.2e}, max |recon scalar diff| = {mr:.2e}; "
          f"dense_ae.npz entries at grid steps vs grid: latent {dz:.2e}, recon {dr:.2e} (see verify.json).", ""]
    for key, v in res.items():
        a = v["agg"]; L += [f"## {key}", f"- test-recon minimum t*: epoch {ci(a['tstar_ep'])}; min value {ci(a['test_min'], 4)}; gap(abs) at t* {ci(a['gap_at_tstar'], 4)}",
                            f"- test recon within 5% of min at epoch {ci(a['test_5pct'])}; within 2%: {ci(a['test_2pct'])}",
                            f"- gap(abs) t25 {ci(a['gap_t25'])}, t50 {ci(a['gap_t50'])}; relative gap te/tr-1 > 10% at epoch {ci(a['gap_rel10'])}",
                            f"- gap vs test-recon derivative xcorr lag (ln t): {ci(a['gap_xcorr_vs_test_lnt'], 2)}", "",
                            f"- gap hinge onset (broken-stick in ln t on [epoch 5, t*]): epoch {ci(a['gap_hinge'])}, R2 {ci(a['gap_hinge_r2'], 2)}", "",
                            "| signature | type | hinge onset (ep) | hinge R2 | hinge detectable | ln(tau/tau_gap) | ln(tau/t*) | value at t* (ref pixel-PCA) | t50 (ep) [frac. of t* value] | ln(t50/gap t50) | xcorr lag vs gap (ln t) | xcorr peak |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in SIGS:
            s = a["sig"][k]; refv = s["ref"]
            L.append(f"| {k} | {'latent-only' if k in LATENT_ONLY else 'DECODER-tied'} | {ci(s['hinge_tau'])} | {ci(s['hinge_r2'], 2)} | {s['hinge_n_det']}/3 | {ci(s['hinge_lnratio_gap'], 2)} | {ci(s['hinge_lnratio_tstar'], 2)} | {ci(s['val_tstar'], 3)} ({f(refv, 3) if refv is not None else '-'}) | {ci(s['t50'])} ({s['n_detectable']}/3) | {ci(s['lnratio_t50_vs_gap'], 2)} | {ci(s['xcorr_lag_vs_gap_lnt'], 2)} | {f(s['xcorr_peak_vs_gap'][0], 2)} |")
        L.append("")
    L += ["## Figures", "fig_timeseries.png (raw series), fig_progress.png (normalised progress, 1 = value at test minimum), fig_onsets.png (t50 per signature), fig_xcorr.png (cross-correlation lags)", ""]
    open(f"{OUT}/RESULTS.md", "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["compute", "report", "verify", "all"]); a = ap.parse_args()
    os.makedirs(f"{OUT}/series", exist_ok=True)
    if a.stage in ("verify", "all"): verify()
    if a.stage in ("compute", "all"):
        for N, wd in CONFIGS:
            for s in SEEDS:
                if os.path.isfile(f"{OUT}/series/{rid(N, wd, s)}.npz"): continue
                print("compute", rid(N, wd, s), flush=True); compute_run(N, wd, s)
    if a.stage in ("report", "all"): report()


if __name__ == "__main__":
    main()
