"""Aggregate vfield_traintest outputs: summary csv + figures.  Usage: python vfield_traintest_aggregate.py [--out DIR]"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import argparse, glob, csv, re, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/traintest"
DEC = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/decomp/summary_runs.csv"; PC = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/perclass/cache"
BANDS = [(1, 10), (10, 100), (100, 1000), (1000, 10 ** 9)]; BN = ["b1_10", "b10_100", "b100_1000", "b1000_up"]


def parse(fn):
    rid = os.path.basename(fn)[:-4]; m = re.search(r"_s(\d+)_(phase|gap|sl40k)_n(\w+?)_(wd|ln)([\d.e-]+)$", rid); seed, fam, N, kind, val = m.groups()
    return rid, int(seed), fam, (60000 if N == "full" else int(N)), (float(val) if kind == "wd" else 0.0), (float(val) if kind == "ln" else 0.0)


def bm(x, st, lo, hi):
    k = (st >= lo) & (st < hi) & np.isfinite(x); return float(np.nanmedian(x[k])) if k.any() else np.nan


def mem_cross(rid):
    p = f"{PC}/v2_{rid}.pkl"
    if not os.path.isfile(p): return np.nan
    P = pickle.load(open(p, "rb")).get("train")
    if P is None or "mem_fit_c" not in P: return np.nan
    st = P["steps"]; out = []
    for c in range(10):
        ok = np.nan_to_num(P["mem_fit_c"][:, c]) > np.nan_to_num(P["mem_cor_c"][:, c])
        if not ok[-1]: continue
        bad = np.where(~ok)[0]; out.append(float(st[0 if len(bad) == 0 else bad[-1] + 1]))
    return float(np.median(out)) if out else np.nan


def series(z):
    """derived series (length K = T-1 unless noted)."""
    T = z["transfer_mean"]; gm = {"sep": 0, "pool": 1}; d = {}
    for g, gi in gm.items():
        for c, ci in (("g", 0), ("c", 1)):
            d[f"gap_{c}_{g}_tgt_te"] = T[gi, 0, 0, ci] - T[gi, 1, 0, ci]      # within(te->te) - cross(tr->te)
            d[f"gap_{c}_{g}_tgt_tr"] = T[gi, 1, 1, ci] - T[gi, 0, 1, ci]      # within(tr->tr) - cross(te->tr)
            d[f"R2{c}_{g}_te_te"] = T[gi, 0, 0, ci]; d[f"R2{c}_{g}_tr_te"] = T[gi, 1, 0, ci]; d[f"R2{c}_{g}_tr_tr"] = T[gi, 1, 1, ci]; d[f"R2{c}_{g}_te_tr"] = T[gi, 0, 1, ci]
        d[f"inc_class_{g}_within_te"] = T[gi, 0, 0, 1] - T[gi, 0, 0, 0]; d[f"inc_class_{g}_within_tr"] = T[gi, 1, 1, 1] - T[gi, 1, 1, 0]
        d[f"inc_class_{g}_cross_tr_to_te"] = T[gi, 1, 0, 1] - T[gi, 1, 0, 0]; d[f"inc_class_{g}_cross_te_to_tr"] = T[gi, 0, 1, 1] - T[gi, 0, 1, 0]
    d["A_excess"] = z["A_rel_dist_cross"] - 0.5 * (z["A_rel_dist_within_te"] + z["A_rel_dist_within_tr"]); d["drift_excess"] = z["drift_rel_dist_cross"] - 0.5 * (z["drift_rel_dist_within_te"] + z["drift_rel_dist_within_tr"])
    d["abscissa_diff"] = z["abscissa_tr"] - z["abscissa_te"]; d["divergence_diff"] = z["divergence_tr"] - z["divergence_te"]
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args(); out = a.out; os.makedirs(f"{out}/figures", exist_ok=True)
    R = []
    for f in sorted(glob.glob(f"{out}/runs/*.npz")):
        rid, seed, fam, N, wd, ln = parse(f); z = np.load(f); R.append(dict(rid=rid, seed=seed, fam=fam, N=N, wd=wd, ln=ln, z=z, st=z["steps"], d=series(z)))
    print(len(R), "runs"); rows = []
    dec = {}
    if os.path.isfile(DEC):
        for r in csv.DictReader(open(DEC)):
            if r["probe"] == "train" and r["family"] == "gap": dec[r["run_id"]] = r
    for r in R:
        z, st, d = r["z"], r["st"], r["d"]; s1 = st[1:]; sm = st[:-1]; row = dict(run_id=r["rid"], family=r["fam"], N=r["N"], wd=r["wd"], ln=r["ln"], seed=r["seed"], n_tr=int(z["n_tr"]), min_class_tr=int(z["min_class_tr"]))
        acc_gap = (z["snap_train_acc"] - z["snap_test_acc"]); loss_gap = z["snap_test_loss"] - z["snap_train_loss"]; ncm_gap = z["ncm_tr_true"] - z["ncm_te"]
        for (lo, hi), bn in zip(BANDS, BN):
            for k, v in d.items(): row[f"{k}_{bn}"] = bm(v, s1, lo, hi)
            row[f"acc_gap_{bn}"] = bm(acc_gap, st, lo, hi); row[f"loss_gap_{bn}"] = bm(loss_gap, st, lo, hi); row[f"ncm_gap_{bn}"] = bm(ncm_gap, st, lo, hi)
            row[f"trratio_log_tr_over_te_{bn}"] = bm(np.log(z["tr_ratio_tr_true"] / z["tr_ratio_te"]), st, lo, hi); row[f"trratio_given_log_tr_over_te_{bn}"] = bm(np.log(z["tr_ratio_tr_given"] / z["tr_ratio_te"]), st, lo, hi)
            if "drag_cos_toM" in z.files:
                row[f"drag_toM_minus_toC_{bn}"] = bm(z["drag_cos_toM"] - z["drag_cos_toC"], s1, lo, hi); row[f"drag_toM_minus_toRand_{bn}"] = bm(z["drag_cos_toM"] - z["drag_cos_toRand"], s1, lo, hi)
        # per-run relation between transfer gap and accuracy gap over time (steps >= 10)
        for key in ("gap_c_sep_tgt_te", "gap_c_pool_tgt_te", "gap_g_sep_tgt_te", "gap_c_sep_tgt_tr"):
            x = d[key]; y = np.interp(s1, st, acc_gap); k = (s1 >= 10) & np.isfinite(x) & np.isfinite(y)
            row[f"spearman_{key}_vs_accgap"] = float(spearmanr(x[k], y[k])[0]) if k.sum() > 10 else np.nan
        # onset of the (smoothed) class-wise transfer gap and of the accuracy gap
        def onset(x, steps, frac=0.5):
            xs = np.convolve(np.nan_to_num(x), np.ones(5) / 5, mode="same"); mx = np.nanmax(xs[5:-5]) if len(xs) > 12 else np.nan
            if not np.isfinite(mx) or mx <= 0: return np.nan
            i = np.where(xs >= frac * mx)[0]; return float(steps[i[0]]) if len(i) else np.nan
        row["onset_transfer_gap_c_te"] = onset(d["gap_c_sep_tgt_te"], s1); row["onset_acc_gap"] = onset(acc_gap, st); row["onset_ncm_gap"] = onset(ncm_gap, st)
        if r["rid"] in dec:
            row["flip_epoch_f1"] = float(dec[r["rid"]]["mis_f1_cross_epoch"]) if dec[r["rid"]]["mis_f1_cross_epoch"] not in ("", "nan") else np.nan
        if r["fam"] == "gap": row["mem_cross_step"] = mem_cross(r["rid"]); row["mem_cross_epoch"] = row["mem_cross_step"] * 128.0 / r["N"]
        if "drag_cos_toM" in z.files:
            dd = z["drag_cos_toM"] - z["drag_cos_toC"]; ep = s1 * 128.0 / r["N"]; pos = np.where(np.nan_to_num(dd) > 0)[0]
            row["drag_frac_positive_ge100"] = float(np.nanmean(dd[s1 >= 100] > 0)) if (s1 >= 100).any() else np.nan
        rows.append(row)
    cols = []
    for d_ in rows:
        for c in d_:
            if c not in cols: cols.append(c)
    with open(f"{out}/summary_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    figures(out, R, rows)
    # dense
    dn = sorted(glob.glob(f"{out}/dense_*.npz"));
    if dn:
        try: dense_figs(out, dn)
        except Exception:
            import traceback; msg = traceback.format_exc(); print("DENSE FIGURE FAILED", msg, file=sys.stderr); open(f"{out}/figures_failed.txt", "a").write(msg)
    print("done")


def figures(out, R, rows):
    fd = f"{out}/figures"; ph = [r for r in R if r["fam"] == "phase"]; sel = [(250, 0.0), (1000, 0.0), (4000, 0.0), (4000, 1e-2), (60000, 0.0), (60000, 3e-2)]
    if os.path.exists(f"{out}/figures_failed.txt"): os.remove(f"{out}/figures_failed.txt")
    try:
        fig, ax = plt.subplots(2, 3, figsize=(17, 8)); ax = ax.ravel()
        for a_, (N, wd) in zip(ax, sel):
            ss = [r for r in ph if r["N"] == N and r["wd"] == wd]
            if not ss: continue
            s1 = ss[0]["st"][1:]
            for key, lab, ls in (("gap_c_sep_tgt_te", "class-wise transfer gap (target test, sep global)", "-"), ("gap_c_pool_tgt_te", "same, pooled global", "--"), ("gap_g_sep_tgt_te", "global-only gap (sep)", ":"), ("gap_c_sep_tgt_tr", "class-wise gap (target train)", "-.")):
                a_.plot(s1, np.nanmedian(np.stack([r["d"][key] for r in ss]), 0), ls=ls, label=lab)
            a2 = a_.twinx(); a2.plot(ss[0]["st"], np.nanmedian(np.stack([r["z"]["snap_train_acc"] - r["z"]["snap_test_acc"] for r in ss]), 0), c="k", lw=.8); a2.set_ylabel("train-test acc gap (eval mode)", fontsize=7)
            a_.set_xscale("log"); a_.set_title(f"N={N} wd={wd:g} (median of {len(ss)} seeds)", fontsize=8); a_.axhline(0, c="gray", lw=.4)
        ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{fd}/transfer_gap_series.png", dpi=60); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        for a_, bn in zip(ax, ("b100_1000", "b1000_up", "b10_100")):
            for fam, mk in (("phase", "o"), ("gap", "^"), ("sl40k", "s")):
                xs = [(x[f"acc_gap_{bn}"], x[f"gap_c_sep_tgt_te_{bn}"], np.log10(x["N"])) for x in rows if x["family"] == fam and np.isfinite(x.get(f"acc_gap_{bn}", np.nan))]
                if xs: a_.scatter([p[0] for p in xs], [p[1] for p in xs], c=[p[2] for p in xs], marker=mk, s=16, cmap="viridis", label=fam)
            a_.set_xlabel(f"train-test accuracy gap ({bn})"); a_.set_ylabel("class-wise transfer gap (target test)"); a_.set_title(f"band {bn}", fontsize=8); a_.axhline(0, c="gray", lw=.4)
        ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{fd}/transfer_vs_accgap.png", dpi=60); plt.close(fig)
        fig, ax = plt.subplots(2, 3, figsize=(17, 8)); ax = ax.ravel()
        for a_, (N, wd) in zip(ax, sel):
            ss = [r for r in ph if r["N"] == N and r["wd"] == wd]
            if not ss: continue
            st = ss[0]["st"]; a_.plot(st, np.nanmedian(np.stack([np.log(r["z"]["tr_ratio_tr_true"] / r["z"]["tr_ratio_te"]) for r in ss]), 0), label="log trace ratio train/test"); a_.set_xscale("log"); a_.axhline(0, c="gray", lw=.4)
            a2 = a_.twinx(); a2.plot(st[1:], np.nanmedian(np.stack([r["d"]["abscissa_diff"] for r in ss]), 0), c="C1", lw=.8); a2.set_yscale("symlog", linthresh=1e-5); a2.set_ylabel("abscissa train - test", fontsize=7, color="C1"); a_.set_title(f"N={N} wd={wd:g}", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{fd}/contraction.png", dpi=60); plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        for k, bn in enumerate(("b100_1000", "b1000_up")):
            for fam, mk in (("phase", "o"), ("gap", "^")):
                xs = [(1.0 / x["N"], x[f"trratio_log_tr_over_te_{bn}"], x["ln"]) for x in rows if x["family"] == fam and np.isfinite(x.get(f"trratio_log_tr_over_te_{bn}", np.nan))]
                if xs: ax[k].scatter([p[0] for p in xs], [p[1] for p in xs], c=[p[2] for p in xs], marker=mk, s=16, cmap="coolwarm")
            ax[k].set_xscale("log"); ax[k].set_xlabel("1/N"); ax[k].set_ylabel("log trace-ratio train/test"); ax[k].set_title(f"{bn} (color = label noise)", fontsize=8); ax[k].axhline(0, c="gray", lw=.4)
        fig.tight_layout(); fig.savefig(f"{fd}/contraction_vs_invN.png", dpi=60); plt.close(fig)
        gp = [r for r in R if r["fam"] == "gap" and "drag_cos_toM" in r["z"].files]
        if gp:
            cm = {250: "C0", 1000: "C1", 4000: "C2", 16000: "C3", 60000: "C4"}; fig, ax = plt.subplots(2, 3, figsize=(17, 8))
            for r in gp:
                ep = r["st"][1:] * 128.0 / r["N"]; z = r["z"]; row = ax[0 if r["ln"] < .3 else 1]
                row[0].plot(ep, z["drag_cos_toM"] - z["drag_cos_toC"], c=cm[r["N"]], lw=.7, label=f"N{r['N']}" if r["seed"] == 0 else None); row[1].plot(ep, z["drag_cos_toM"] - z["drag_cos_toRand"], c=cm[r["N"]], lw=.7); row[2].plot(ep, z["drag_cosM_drift_toC"] - z["drag_cosM_drift_toTrue"], c=cm[r["N"]], lw=.7)
            for k in (0, 1):
                for j in range(3): ax[k, j].set_xscale("log"); ax[k, j].axhline(0, c="k", lw=.4)
                ax[k, 0].set_title(f"cos(test class drift -> mislabelled mean) - cos(-> clean mean), ln={0.2 if k==0 else 0.4}", fontsize=8); ax[k, 1].set_title("... minus cos(-> random clean subset)", fontsize=8); ax[k, 2].set_title("mislabelled drift: cos(-> given-class clean mean) - cos(-> true-class)", fontsize=8)
            for r in gp:
                x = next((q for q in rows if q["run_id"] == r["rid"]), None)
                if x is None: continue
                for k in (0, 1):
                    if (k == 0) == (r["ln"] < .3):
                        for j in range(3):
                            if np.isfinite(x.get("mem_cross_epoch", np.nan)): ax[k, j].axvline(x["mem_cross_epoch"], c=cm[r["N"]], ls=":", lw=.5)
            ax[0, 0].legend(fontsize=7); fig.suptitle("dotted vertical lines: NCM memorisation crossover epoch of each run", fontsize=8); fig.tight_layout(); fig.savefig(f"{fd}/label_noise_drag.png", dpi=60); plt.close(fig)
    except Exception:
        import traceback; msg = traceback.format_exc(); print("FIGURE FAILED", msg, file=sys.stderr); open(f"{out}/figures_failed.txt", "a").write(msg)


def dense_figs(out, files):
    fd = f"{out}/figures"; fig, ax = plt.subplots(len(files), 4, figsize=(20, 4.5 * len(files)), squeeze=False)
    for i, f in enumerate(files):
        z = np.load(f); cells = z["cells"]; ts = np.unique(cells[:, 0]); Ls = np.unique(cells[:, 1])
        for j, frame in enumerate(("latent", "logit")):
            T = z[f"{frame}_transfer"]; gap = T[0, 0, 0, 1] - T[0, 1, 0, 1]; safe = (z[f"{frame}_floor_ratio_te"] < 0.1) & (z[f"{frame}_floor_ratio_tr"] < 0.1)
            M = np.full((len(Ls), len(ts)), np.nan)
            for k, (t, L) in enumerate(cells):
                if safe[k]: M[np.where(Ls == L)[0][0], np.where(ts == t)[0][0]] = gap[k]
            im = ax[i, j].imshow(M, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.3, vmax=0.3); ax[i, j].set_title(f"{os.path.basename(f)[6:-4]} {frame}: class-wise transfer gap (te target), floor-safe only", fontsize=7)
            ax[i, j].set_xticks(range(0, len(ts), 4)); ax[i, j].set_xticklabels([str(t) for t in ts[::4]], fontsize=6, rotation=60); ax[i, j].set_yticks(range(len(Ls))); ax[i, j].set_yticklabels(Ls, fontsize=6); plt.colorbar(im, ax=ax[i, j], fraction=.046)
            ax[i, j].set_xlabel("t"); ax[i, j].set_ylabel("lag")
        for j, frame in enumerate(("latent", "logit")):
            T = z[f"{frame}_transfer"]; safe = (z[f"{frame}_floor_ratio_te"] < 0.1) & (z[f"{frame}_floor_ratio_tr"] < 0.1)
            for L in Ls:
                k = np.where((cells[:, 1] == L) & safe)[0]
                if len(k): ax[i, 2 + j].plot(cells[k, 0], (T[0, 0, 0, 1] - T[0, 1, 0, 1])[k], marker="o", ms=2, label=f"lag {L}")
            ax[i, 2 + j].set_xscale("log"); ax[i, 2 + j].set_title(f"{frame}: transfer gap vs t at each lag (floor-safe)", fontsize=7); ax[i, 2 + j].axhline(0, c="gray", lw=.4)
        ax[i, 2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{fd}/dense_lags.png", dpi=55); plt.close(fig)


if __name__ == "__main__": main()
