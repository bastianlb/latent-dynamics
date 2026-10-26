"""Null/power analysis of the train-minus-test global-affine residual share (aff_diff) on dense AE runs.  CPU only, <=8 threads.
Usage: python ae_dense_null.py [--stage compute|report|all]
Does NOT touch the outputs of ae_dense_leadlag.py (reads dense runs, writes analysis/null_series/*.npz, results_null.json, fig_null_*.png, RESULTS_null.md).
Per snapshot t (velocity v = z(t+W)-z(t-W), W = 12 steps for ALL runs so every-2 and every-4 runs are comparable; the
original leadlag uses W=3 snapshots = 6 steps for every-2 runs):
  aff_tr, aff_te : adjusted residual share of the global affine fit v ~ [z,1] on the train / test probe (1000 pts each)
  aff_diff       = aff_tr - aff_te
  floor_*        : test-vs-test split-half baseline: random disjoint halves of the test probe (size min(n_train_probe, 500)), aff(A)-aff(B); K=8 fixed splits ->
                   mean and sd over splits (sd = noise floor of a train-test difference at n=500 each; train-test uses n=1000 each
                   so the floor is conservative); also train-vs-train split halves.
  gap            = recon_test - recon_train (full-pool / full-test recon).
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import argparse, json
import numpy as np
from scipy.ndimage import median_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ae_dense_leadlag import aff_resid, boot_ci

DENSE = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/dense"; OUT = f"{DENSE}/analysis"
WSTEPS = 12; K = 8
CFG = [(60000, "full"), (4000, "4000"), (1000, "1000"), (250, "250")]
SEEDS = {60000: (0, 1, 2), 4000: (0, 1, 2), 1000: (0, 1, 2, 3, 4, 5), 250: (0, 1, 2, 3, 4, 5)}
def rid(N, s): return f"mnist_ae_mlp_d16_s{s}_n{N}_wd0"
def steps_per_epoch(N): return N / 128.0


def compute(N, s):
    D = np.load(f"{DENSE}/runs/{rid(N, s)}/dense_ae.npz"); st = D["step"].astype(int); ev = int(st[1] - st[0]); W = WSTEPS // ev
    Zte, Ztr = D["z_test"].astype(np.float64), D["z_train"].astype(np.float64); T = len(st)
    rs = np.random.RandomState(123); n = Zte.shape[1]; h = min(Ztr.shape[1], n // 2)  # half size matched to the train probe size when smaller (N=250)
    sp_te = [rs.permutation(n) for _ in range(K)]; sp_tr = [rs.permutation(Ztr.shape[1]) for _ in range(K)]; htr = Ztr.shape[1] // 2
    keys = ["aff_tr", "aff_te", "aff_diff", "fl_te_mean", "fl_te_sd", "fl_tr_mean", "fl_tr_sd", "fl_te_abs"]
    R = {k: np.full(T, np.nan) for k in keys}
    for t in range(W, T - W):
        vt, ve = Ztr[t + W] - Ztr[t - W], Zte[t + W] - Zte[t - W]
        a, b = aff_resid(Ztr[t], vt), aff_resid(Zte[t], ve); R["aff_tr"][t], R["aff_te"][t], R["aff_diff"][t] = a, b, a - b
        d = [aff_resid(Zte[t][p[:h]], ve[p[:h]]) - aff_resid(Zte[t][p[h:2 * h]], ve[p[h:2 * h]]) for p in sp_te]
        R["fl_te_mean"][t], R["fl_te_sd"][t], R["fl_te_abs"][t] = np.mean(d), np.std(d, ddof=1), np.mean(np.abs(d))
        d = [aff_resid(Ztr[t][p[:htr]], vt[p[:htr]]) - aff_resid(Ztr[t][p[htr:2 * htr]], vt[p[htr:2 * htr]]) for p in sp_tr]
        R["fl_tr_mean"][t], R["fl_tr_sd"][t] = np.mean(d), np.std(d, ddof=1)
    np.savez(f"{OUT}/null_series/{rid(N, s)}.npz", steps=st, N=N, seed=s, ev=ev,
             recon_train=D["recon_train"], recon_test=D["recon_test"], **R)


def sm(x, k=9): 
    y = x.copy(); ok = np.isfinite(x); y[ok] = median_filter(x[ok], size=k, mode="nearest"); return y


def load(N, s): return dict(np.load(f"{OUT}/null_series/{rid(N, s)}.npz"))


def interp_at(S, x, step):
    ok = np.isfinite(x); return float(np.interp(step, S["steps"][ok], sm(x)[ok]))


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = {}; EP = [2, 4, 6, 10, 14]; ST = [200, 500, 1000, 2000, 4000]
    for N, nm in CFG:
        per = []
        for s in SEEDS[N]:
            S = load(N, s); spe = steps_per_epoch(N); ep = S["steps"] / spe
            gap = sm(S["recon_test"] - S["recon_train"]); fl = sm(S["fl_te_sd"]); ok = np.isfinite(S["aff_diff"])
            r = dict(seed=s, **{f"aff_diff_ep{e}": interp_at(S, S["aff_diff"], e * spe) for e in EP if e * spe <= S["steps"][-1]},
                     **{f"aff_diff_step{k}": interp_at(S, S["aff_diff"], k) for k in ST if k <= S["steps"][-1]},
                     **{f"aff_tr_ep{e}": interp_at(S, S["aff_tr"], e * spe) for e in EP if e * spe <= S["steps"][-1]},
                     **{f"aff_te_ep{e}": interp_at(S, S["aff_te"], e * spe) for e in EP if e * spe <= S["steps"][-1]},
                     **{f"gap_ep{e}": float(np.interp(e * spe, S["steps"], gap)) for e in EP if e * spe <= S["steps"][-1]},
                     gap_end=float(gap[-1]), gap_max=float(gap.max()), step_end=int(S["steps"][-1]), ep_end=float(ep[-1]),
                     floor_sd_med=float(np.nanmedian(S["fl_te_sd"])), floor_trtr_sd_med=float(np.nanmedian(S["fl_tr_sd"])),
                     floor_mean_med=float(np.nanmedian(S["fl_te_mean"])), aff_diff_max=float(np.nanmax(sm(S["aff_diff"]))), aff_diff_end=float(sm(S["aff_diff"])[ok][-1]))
            # onset: final crossing of aff_diff above floor_mean + 3*sd_floor (smoothed, window 9)
            ad = sm(S["aff_diff"]); thr = np.nanmedian(S["fl_te_mean"]) + 3 * np.nanmedian(S["fl_te_sd"]); below = np.where((ad < thr) & ok)[0]
            first = np.where(ok & (ad >= thr))[0]
            r["thr"] = float(thr); r["onset_final_cross_step"] = float(S["steps"][min(below[-1] + 1, len(ad) - 1)]) if len(below) else float(S["steps"][first[0]]) if len(first) else np.nan
            r["onset_first_cross_step"] = float(S["steps"][first[0]]) if len(first) else np.nan
            r["onset_first_cross_ep"] = r["onset_first_cross_step"] / spe
            # peak
            i = int(np.nanargmax(ad)); r["peak_aff_diff"] = float(ad[i]); r["peak_step"] = float(S["steps"][i]); r["peak_ep"] = float(ep[i])
            # gap onset (rel gap > 10%) in steps
            rel = S["recon_test"] / S["recon_train"] - 1; j = np.where(sm(rel) >= 0.10)[0]; r["gap10_step"] = float(S["steps"][j[0]]) if len(j) else np.nan
            r["gap10_ep"] = r["gap10_step"] / spe if np.isfinite(r["gap10_step"]) else np.nan
            per.append(r)
        res[nm] = dict(N=N, per_seed=per)
    json.dump(res, open(f"{OUT}/results_null.json", "w"), indent=1, default=float)
    # ---------------- figures
    colr = {"full": "C3", "4000": "C1", "1000": "C0", "250": "C2"}
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    for N, nm in CFG:
        for s in SEEDS[N]:
            S = load(N, s); spe = steps_per_epoch(N); ep = S["steps"] / spe; lab = f"N={nm}" if s == 0 else None
            for a, x in ((ax[0, 0], S["steps"]), (ax[0, 1], ep)):
                a.plot(x, sm(S["aff_diff"]), colr[nm], alpha=.5, lw=.8, label=lab)
            ax[0, 2].plot(ep, sm(S["recon_test"] - S["recon_train"]), colr[nm], alpha=.5, lw=.8, label=lab)
            ax[1, 0].plot(ep, sm(S["aff_tr"]), colr[nm], alpha=.5, lw=.8, label=lab); ax[1, 1].plot(ep, sm(S["aff_te"]), colr[nm], alpha=.5, lw=.8)
            ax[1, 2].plot(S["steps"], sm(S["aff_diff"]) / np.nanmedian(S["fl_te_sd"]), colr[nm], alpha=.5, lw=.8)
    for a in ax.flat: a.set_xscale("log")
    for a, t, xl in ((ax[0, 0], "aff_diff (train-test) vs STEP", "step"), (ax[0, 1], "aff_diff vs EPOCH (per-point exposure)", "epoch"), (ax[0, 2], "recon gap vs epoch", "epoch"),
                     (ax[1, 0], "aff_tr (train residual share)", "epoch"), (ax[1, 1], "aff_te (test residual share)", "epoch"), (ax[1, 2], "aff_diff / floor sd (z-score vs split-half)", "step")):
        a.set_title(t); a.set_xlabel(xl)
    ax[0, 2].set_yscale("log"); ax[0, 0].axhline(0, color="k", lw=.5); ax[0, 1].axhline(0, color="k", lw=.5); ax[1, 2].axhline(3, color="k", ls=":", lw=.6); ax[0, 0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig_null_affdiff.png", dpi=105); plt.close(fig)
    # floor figure: one N=1000 and one full run
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, (N, nm) in zip(ax, CFG):
        S = load(N, 0); ep = S["steps"] / steps_per_epoch(N)
        a.plot(ep, sm(S["aff_diff"]), "k", label="train - test"); a.plot(ep, sm(S["fl_te_mean"]), "C0", label="test half - test half (mean of 8 splits)")
        a.fill_between(ep, sm(S["fl_te_mean"]) - 2 * sm(S["fl_te_sd"]), sm(S["fl_te_mean"]) + 2 * sm(S["fl_te_sd"]), color="C0", alpha=.25, label="+-2 sd over splits")
        a.plot(ep, sm(S["fl_tr_mean"]), "C1", lw=.8, label="train half - train half"); a.set_xscale("log"); a.set_title(f"N={nm} seed0"); a.set_xlabel("epoch")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig_null_floor.png", dpi=105); plt.close(fig)
    # amplitude vs eventual gap across N
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for N, nm in CFG:
        P = res[nm]["per_seed"]; g = [p["gap_end"] if nm in ("full", "4000") else p["gap_max"] for p in P]
        for key, a in (("aff_diff_ep6", ax[0]), ("peak_aff_diff", ax[1])):
            v = [p.get(key, np.nan) for p in P]; a.scatter(g, v, c=colr[nm], label=f"N={nm}")
    ax[0].set_xlabel("max recon gap in window"); ax[0].set_ylabel("aff_diff at epoch 6"); ax[1].set_xlabel("max recon gap in window"); ax[1].set_ylabel("peak aff_diff"); ax[0].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_null_vs_gap.png", dpi=105); plt.close(fig)
    return res


def ms(vals, p=4):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    return "n/a" if not len(v) else f"{v.mean():.{p}f} +- {v.std(ddof=1) if len(v) > 1 else 0:.{p}f}"


def write_md(res):
    L = ["# Null / power test of the train-minus-test affine-residual signature", "", open(f"{OUT}/_null_verdict.txt").read() if os.path.exists(f"{OUT}/_null_verdict.txt") else "", "",
         "Definitions in ae_dense_null.py docstring. aff_diff = adj. residual share of global affine fit of latent velocity (train probe) minus (test probe); W = 12 steps for all runs (NOT identical to the W=6 steps of RESULTS.md, so numbers differ slightly). All entries seed mean +- sd (n=3 for full/4000, n=6 for 1000/250 = seeds 0-5). Epoch = step*128/N.", ""]
    L += ["## aff_diff at fixed EPOCH of per-point exposure", "| N | " + " | ".join(f"ep {e}" for e in (2, 4, 6, 10, 14)) + " | peak | peak epoch | first crossing of floor+3sd (epoch) |", "|---|" + "---|" * 8]
    for nm, v in res.items():
        P = v["per_seed"]; L.append(f"| {nm} | " + " | ".join(ms([p.get(f"aff_diff_ep{e}") for p in P]) for e in (2, 4, 6, 10, 14)) + f" | {ms([p['peak_aff_diff'] for p in P])} | {ms([p['peak_ep'] for p in P], 1)} | {ms([p['onset_first_cross_ep'] for p in P], 1)} |")
    L += ["", "## aff_diff at fixed STEP", "| N | " + " | ".join(f"step {k}" for k in (200, 500, 1000, 2000, 4000)) + " |", "|---|" + "---|" * 5]
    for nm, v in res.items():
        P = v["per_seed"]; L.append(f"| {nm} | " + " | ".join(ms([p.get(f"aff_diff_step{k}") for p in P]) for k in (200, 500, 1000, 2000, 4000)) + " |")
    L += ["", "## Noise floors and gap", "| N | split-half test-test sd (median over time) | split-half train-train sd | median bias test-test | threshold floor+3sd | recon gap at epoch 6 | gap at end of window | rel gap>10% at epoch |", "|---|---|---|---|---|---|---|---|"]
    for nm, v in res.items():
        P = v["per_seed"]; L.append(f"| {nm} | {ms([p['floor_sd_med'] for p in P])} | {ms([p['floor_trtr_sd_med'] for p in P])} | {ms([p['floor_mean_med'] for p in P])} | {ms([p['thr'] for p in P])} | {ms([p.get('gap_ep6') for p in P])} | {ms([p['gap_end'] for p in P])} | {ms([p['gap10_ep'] for p in P], 1)} |")
    L += ["", "## Train-only and test-only residual share (epoch 6)", "| N | aff_tr | aff_te |", "|---|---|---|"]
    for nm, v in res.items():
        P = v["per_seed"]; L.append(f"| {nm} | {ms([p.get('aff_tr_ep6') for p in P])} | {ms([p.get('aff_te_ep6') for p in P])} |")
    L += ["", "## Figures", "fig_null_affdiff.png (aff_diff vs step / epoch, gap, train and test parts, z-score), fig_null_floor.png (split-half floor), fig_null_vs_gap.png (amplitude vs gap)"]
    open(f"{OUT}/RESULTS_null.md", "w").write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); ap.add_argument("--only", default=None); a = ap.parse_args()
    os.makedirs(f"{OUT}/null_series", exist_ok=True)
    if a.stage in ("compute", "all"):
        for N, nm in CFG:
            for s in SEEDS[N]:
                if os.path.isfile(f"{OUT}/null_series/{rid(N, s)}.npz") or not os.path.isfile(f"{DENSE}/runs/{rid(N, s)}/model_final.pt"): continue
                if a.only and a.only != nm: continue
                print("compute", rid(N, s), flush=True); compute(N, s)
    if a.stage in ("report", "all"): write_md(report())


def extra():
    """window-mean tables: aff_diff averaged over epoch windows vs split-half floor of the same windowed average; appended to RESULTS_null.md"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    wins = [(2, 6), (6, 14), (14, 30), (30, 60), (60, 120), (120, 250)]; tab = {}; fl = {w: [] for w in wins}
    for N, nm in CFG:
        for s in SEEDS[N]:
            S = load(N, s); ep = S["steps"] / steps_per_epoch(N)
            for w in wins:
                m = (ep >= w[0]) & (ep < w[1]) & np.isfinite(S["aff_diff"])
                if m.sum() < 3: continue
                tab.setdefault((nm, w), []).append((np.mean(S["aff_diff"][m]), np.mean(S["fl_te_mean"][m]), np.mean(S["aff_tr"][m] - 0 * S["aff_te"][m]), np.mean(S["recon_test"][m] - S["recon_train"][m])))
                fl[w].append(np.mean(S["fl_te_mean"][m]))
    L = ["", "## Window-averaged aff_diff (mean over snapshots in the epoch window) vs floor", "Floor column: sd across all runs/seeds (all N pooled) of the same window-average of the test-vs-test split-half difference (no train-test asymmetry, so any spread is pure noise); 'z' = (seed-mean aff_diff - mean floor) / floor-sd.", "",
         "| window (epoch) | floor sd (windowed) | " + " | ".join(nm for _, nm in CFG) + " |", "|---|---|" + "---|" * 4]
    for w in wins:
        fsd = np.std(fl[w], ddof=1) if len(fl[w]) > 2 else np.nan; row = f"| {w[0]}-{w[1]} | {fsd:.4f} | "; cells = []
        for N, nm in CFG:
            v = tab.get((nm, w))
            if not v: cells.append("n/a"); continue
            a = np.array([x[0] for x in v]); fm = np.mean([x[1] for x in v]); cells.append(f"{a.mean():.4f} +- {a.std(ddof=1):.4f} (z={(a.mean() - fm) / fsd:.1f}; mean gap {np.mean([x[3] for x in v]):.3f}; n={len(a)})")
        L.append(row + " | ".join(cells) + " |")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)); colr = {"full": "C3", "4000": "C1", "1000": "C0", "250": "C2"}
    for N, nm in CFG:
        xs, ys, es, gs = [], [], [], []
        for w in wins:
            v = tab.get((nm, w))
            if v: a = np.array([x[0] for x in v]); xs.append(np.sqrt(w[0] * w[1])); ys.append(a.mean()); es.append(a.std(ddof=1)); gs.append(np.mean([x[3] for x in v]))
        ax[0].errorbar(xs, ys, es, color=colr[nm], marker="o", label=f"N={nm}", capsize=2); ax[1].plot(gs, ys, color=colr[nm], marker="o", label=f"N={nm}")
    fsd = np.mean([np.std(fl[w], ddof=1) for w in wins if len(fl[w]) > 2]); ax[0].axhspan(-2 * fsd, 2 * fsd, color="gray", alpha=.25, label="+-2 floor sd (windowed)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("epoch (window centre)"); ax[0].set_ylabel("aff_diff (window mean, seed mean +- sd)"); ax[0].legend(fontsize=7)
    ax[1].set_xscale("symlog", linthresh=.01); ax[1].set_xlabel("mean recon gap in window"); ax[1].set_ylabel("aff_diff"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_null_windows.png", dpi=105)
    open(f"{OUT}/RESULTS_null.md", "a").write("\n".join(L) + "\n\nfig_null_windows.png (window means vs epoch with floor band; aff_diff vs gap)\n")
