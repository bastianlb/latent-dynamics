from latentdyn._threads import cap_threads; cap_threads(4)
"""Regime sweep of non-topological latent metrics (plus topology for reference): classifier, MLP AE, linear AE.
Reads geometry caches (ae_dynamics/analysis/geom), topology caches (ae_dynamics/analysis/topo, analysis_v2/cache), regime_numbers.json.
Writes figures to dynamics_report/figures/ (regime_metrics_full.png, regime_metrics, regime_traintest, regime_acrossN) and table + takeaways inputs to analysis/geom/.
Usage: python scripts/ae/ae_regime_metrics_fig.py [full|curated|traintest|acrossN|table|all]"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

BV = "/vol/bitbucket/lbastian/latent-dynamics"; GEOM = f"{BV}/ae_dynamics/analysis/geom"; TOPO = f"{BV}/ae_dynamics/analysis/topo"
CACHE = f"{BV}/analysis_v2/cache"; OUT = f"{BV}/dynamics_report/figures"
plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.titlesize": 7, "lines.linewidth": 0.9})
CL, AE, LIN = "#d62728", "#1f77b4", "#7f7f7f"
DS = ("mnist", "fashion_mnist"); SEEDS = (0, 1, 2); MODELS = ("clf", "mlp", "lin")
COL = {"clf": CL, "mlp": AE, "lin": LIN}; NAME = {"clf": "classifier", "mlp": "MLP AE", "lin": "linear AE"}
LS = {"mnist": "-", "fashion_mnist": "--"}; DL = {"mnist": "MNIST", "fashion_mnist": "FM"}
BANDS = (("early fit", "#fff2a8"), ("fitting", "#b9e6c0"), ("overfitting", "#d9ccf0"))
G0 = 33.458; SPANS = ((0.125, 1.0), (1.0, G0), (G0, 2400.0))
NS = ("250", "1000", "4000", "full"); NNUM = {"250": 250, "1000": 1000, "4000": 4000, "full": 60000}
GEO = ["mpd_raw", "norm_mean", "pr_dim", "top2_var", "twonn_dim", "local_pr_dim", "knn_label_acc", "knn_pix_acc", "ncm_label_acc", "ncm_pix_acc",
       "fisher_label", "fisher_pix", "sil_label", "sil_pix", "kmeans_nmi_label", "kmeans_nmi_pix", "knn_overlap_pixel", "knn_overlap_prev",
       "knn_overlap_init", "knn_dist_cv", "hubness_skew", "nn1_ratio", "rho_err_knn"]
CROSS = ["test_to_train_nn", "train_to_test_nn", "rho_test_err_train_nn", "pr_dim_diff"]
TOPOM = ["h1_entropy", "h1_total", "pmag_fixed"]
LAB = {"mpd_raw": "mean pairwise distance (raw)", "norm_mean": "mean norm (raw)", "pr_dim": "participation ratio", "top2_var": "top-2 variance share",
       "twonn_dim": "TwoNN dimension", "local_pr_dim": "local PR dimension", "knn_label_acc": "kNN label accuracy", "knn_pix_acc": "kNN pixel-cluster accuracy",
       "ncm_label_acc": "NCM label accuracy", "ncm_pix_acc": "NCM pixel-cluster accuracy", "fisher_label": "Fisher ratio, label", "fisher_pix": "Fisher ratio, pixel cluster",
       "sil_label": "silhouette, label", "sil_pix": "silhouette, pixel cluster", "kmeans_nmi_label": "k-means NMI, label", "kmeans_nmi_pix": "k-means NMI, pixel cluster",
       "knn_overlap_pixel": "kNN overlap with pixel kNN", "knn_overlap_prev": "kNN overlap with previous snapshot", "knn_overlap_init": "kNN overlap with step 0",
       "knn_dist_cv": "kNN distance CV", "hubness_skew": "hubness skew", "nn1_ratio": "1-NN distance / mean distance", "rho_err_knn": "Spearman(recon error, kNN dist)",
       "test_to_train_nn": "test-to-train NN distance", "train_to_test_nn": "train-to-test NN distance", "rho_test_err_train_nn": "Spearman(err, test-to-train NN)",
       "pr_dim_diff": "PR dim difference", "h1_entropy": "H1 persistence entropy", "h1_total": "H1 lifetime sum", "pmag_fixed": "PMag", "gap": "final gap, test/train recon - 1"}


def rid(m, ds, s, N, wd):
    return f"{ds}_probe_none_d16_s{s}_phase_n{N}_wd0" if m == "clf" else f"{ds}_ae_{'mlp' if m == 'mlp' else 'linear'}_d16_s{s}_n{N}_wd{wd}"


def load_run(m, ds, s, N, wd):
    r = rid(m, ds, s, N, wd); f = f"{GEOM}/{r}.npz"
    if not os.path.isfile(f): return None
    z = np.load(f); d = {"step": z["step"], "epoch": z["run/epoch"]}
    for k in z.files:
        if k.startswith(("run/", "test/", "train/", "cross/")): d[k] = z[k]
    ft = f"{CACHE}/{r}.npz" if m == "clf" else f"{TOPO}/{r}.npz"
    if os.path.isfile(ft):
        t = np.load(ft)
        for k in TOPOM:
            if f"test/{k}" in t.files: d[f"topo_test/{k}"] = t[f"test/{k}"]
            if m == "clf" and f"train/{k}" in t.files: d[f"topo_train/{k}"] = t[f"train/{k}"]
    if m != "clf":
        ft = f"{TOPO}/{r}__train.npz"
        if os.path.isfile(ft):
            t = np.load(ft)
            for k in TOPOM:
                if f"train/{k}" in t.files: d[f"topo_train/{k}"] = t[f"train/{k}"]
    d["gap"] = d["run/recon_test"] / d["run/recon_train"] - 1 if "run/recon_test" in d else np.full(120, np.nan)
    return d


def load_all():
    """R[(model, ds, N, wd)] -> list over seeds of run dicts (None if missing)."""
    R = {}
    for ds in DS:
        for s_ in [0]:
            for m in MODELS:
                for N in NS:
                    for wd in ("0", "1e-4"):
                        if m == "clf" and (N != "1000" or wd != "0"): continue
                        if m == "lin" and wd != "0": continue
                        runs = [load_run(m, ds, s, N, wd) for s in SEEDS]
                        if all(r is not None for r in runs): R[(m, ds, N, wd)] = runs
    return R


def metric_key(probe, name):
    """probe in {test, train, cross, topo_test, topo_train, run}."""
    return name if probe == "run" and name == "gap" else f"{probe}/{name}"


def series(R, m, ds, N, wd, key):
    """(epoch, mean, sd) over seeds, final snapshot dropped; None if missing."""
    runs = R.get((m, ds, N, wd))
    if runs is None or any(key not in r for r in runs): return None
    ep = np.mean([r["epoch"] for r in runs], 0)[:-1]; A = np.array([r[key] for r in runs], float)[:, :-1]
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore")
        return ep, np.nanmean(A, 0), np.nanstd(A, 0)


def lighten(c, f=0.55): return tuple(1 - (1 - v) * (1 - f) for v in matplotlib.colors.to_rgb(c))


def style(ax):
    ax.set_xscale("log"); ax.grid(alpha=0.25, lw=0.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ax.spines.values(): s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2, pad=1.5, labelsize=6)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6)); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def bands(ax, spans=SPANS):
    for (lo, hi), (_, c) in zip(spans, BANDS): ax.axvspan(lo, hi, color=c, alpha=0.35, lw=0, zorder=0)
    ax.axvline(12.5, color="0.25", ls=":", lw=0.8, zorder=1); ax.set_xlim(0.125, 2400)


def band_labels(ax, fs=6):
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for (lo, hi), (nm, _) in zip(SPANS, BANDS): ax.text(np.sqrt(lo * hi), 1.02, nm, transform=tr, ha="center", va="bottom", fontsize=fs, color="0.25")


def draw(ax, R, key, N="1000", wd="0", models=MODELS, lw=0.9, col=None, alpha=0.15, dss=DS):
    for ds in dss:
        for m in models:
            s = series(R, m, ds, N, wd, key)
            if s is None: continue
            ep, mu, sd = s; ok = np.isfinite(mu) & (ep > 0)
            if ok.sum() < 2: continue
            c = (col or {}).get(m, COL[m]) if isinstance(col, dict) else (col or COL[m])
            ax.plot(ep[ok], mu[ok], color=c, ls=LS[ds], lw=lw)
            ax.fill_between(ep[ok], (mu - sd)[ok], (mu + sd)[ok], color=c, alpha=alpha, lw=0)


def shared_legend(fig, ncol=3, y=None):
    h = [Patch(facecolor=c, alpha=0.35, lw=0) for _, c in BANDS] + [Line2D([], [], color=COL[m], lw=1) for m in MODELS] + [Line2D([], [], color="0.3", ls=LS[d], lw=1) for d in DS] + [Line2D([], [], color="0.25", ls=":", lw=0.8)]
    lab = ["early fit, <1 epoch", "fitting, 1 to 33 epochs", "overfitting, >33 epochs"] + [NAME[m] for m in MODELS] + ["MNIST", "FM", "step 100 (12.5 epochs)"]
    fig.legend(h, lab, loc="lower center", ncol=ncol, fontsize=7, frameon=False, handlelength=1.8, columnspacing=1.5, labelspacing=0.3)


# ---------------------------------------------------------------- figures
def full_fig(R):
    items = [("test", g) for g in GEO] + [("train", g) for g in GEO] + [("cross", c) for c in CROSS] + [("topo_test", t) for t in TOPOM] + [("topo_train", t) for t in TOPOM]
    nc = 6; nr = int(np.ceil(len(items) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(15, 2.0 * nr), squeeze=False)
    for ax, (p, k) in zip(axs.ravel(), items):
        style(ax); bands(ax); draw(ax, R, metric_key(p, k)); ax.set_title(f"{p}: {LAB[k]}", fontsize=7)
    for ax in axs.ravel()[len(items):]: ax.axis("off")
    for ax in axs[-1]: ax.set_xlabel("epoch", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); shared_legend(fig, ncol=9)
    fig.savefig(f"{OUT}/regime_metrics_full.png", dpi=300); plt.close(fig)


CUR = [("test", "pr_dim"), ("test", "twonn_dim"), ("test", "local_pr_dim"), ("test", "top2_var"),
       ("test", "knn_label_acc"), ("test", "knn_pix_acc"), ("test", "knn_overlap_pixel"), ("test", "fisher_label"),
       ("test", "knn_overlap_prev"), ("test", "knn_overlap_init"), ("test", "knn_dist_cv"), ("test", "hubness_skew"),
       ("test", "rho_err_knn"), ("cross", "test_to_train_nn"), ("topo_test", "h1_total"), ("topo_test", "pmag_fixed")]


def curated(R):
    fig, axs = plt.subplots(4, 4, figsize=(7.2, 6.4))
    for i, (ax, (p, k)) in enumerate(zip(axs.ravel(), CUR)):
        style(ax); bands(ax); draw(ax, R, metric_key(p, k)); ax.set_title(LAB[k], fontsize=7, pad=9 if i < 4 else 3)
        if i < 4: band_labels(ax, 5.5)
        if i >= 12: ax.set_xlabel("epoch", fontsize=7, labelpad=1)
        if k == "hubness_skew" or k.endswith("_nn") or k == "twonn_dim": pass
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.94, wspace=0.34, hspace=0.45); shared_legend(fig, ncol=4)
    for e in ("pdf", "png"): fig.savefig(f"{OUT}/regime_metrics.{e}", dpi=300)
    plt.close(fig)


def markers_for(R, N):
    """gap onset (>10%) and test-min epoch of MLP AE wd0 at N, mean over seeds & datasets; also per ds."""
    out = {}
    for ds in DS:
        g, mn = [], []
        for r in R[("mlp", ds, N, "0")]:
            k = np.where(r["gap"][:-1] > 0.10)[0]; g.append(r["epoch"][k[0]] if len(k) else np.nan); mn.append(r["epoch"][:-1][np.nanargmin(r["run/recon_test"][:-1])])
        out[ds] = (float(np.nanmean(g)) if np.isfinite(g).any() else np.nan, float(np.nanmean(mn)))
    return out


TT = [("pr_dim", "participation ratio"), ("knn_dist_cv", "kNN distance CV"), ("rho_err_knn", "Spearman(err, kNN dist)"),
      ("h1_total", "H1 lifetime sum"), ("cross", "cross-probe NN distance"), ("knn_overlap_pixel", "kNN overlap, pixel")]


def traintest(R):
    fig, axs = plt.subplots(2, 6, figsize=(7.2, 3.9))
    tint = lighten(AE)
    for ri, N in enumerate(("1000", "full")):
        mk = markers_for(R, N)
        for ci, (k, t) in enumerate(TT):
            ax = axs[ri, ci]; style(ax)
            if N == "1000": bands(ax)
            else: ax.set_xlim(0.125, 60); ax.axvline(12.5, color="0.25", ls=":", lw=0.8, zorder=1)
            for ds in DS:
                if k == "cross":
                    for kk, c in (("test_to_train_nn", AE), ("train_to_test_nn", tint)):
                        s = series(R, "mlp", ds, N, "0", f"cross/{kk}")
                        if s: ok = np.isfinite(s[1]) & (s[0] > 0); ax.plot(s[0][ok], s[1][ok], color=c, ls=LS[ds], lw=0.9)
                    continue
                pt, pr = ("topo_test", "topo_train") if k == "h1_total" else ("test", "train")
                for pp, c in ((pr, tint), (pt, AE)):
                    s = series(R, "mlp", ds, N, "0", f"{pp}/{k}")
                    if s is None: continue
                    ep, mu, sd = s; ok = np.isfinite(mu) & (ep > 0)
                    ax.plot(ep[ok], mu[ok], color=c, ls=LS[ds], lw=0.9); ax.fill_between(ep[ok], (mu - sd)[ok], (mu + sd)[ok], color=c, alpha=0.15, lw=0)
            gs = [mk[d][0] for d in DS if np.isfinite(mk[d][0])]
            if gs: ax.axvline(np.mean(gs), color="k", ls="-.", lw=0.6)
            ax.axvline(np.mean([mk[d][1] for d in DS]), color="k", ls=(0, (1, 1)), lw=0.9)
            ax.set_title(t, fontsize=6.3, pad=3)
            if ri == 1: ax.set_xlabel("epoch", fontsize=7, labelpad=1)
        axs[ri, 0].set_ylabel(f"N = {NNUM[N]}", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.27, top=0.93, wspace=0.42, hspace=0.35)
    h = [Patch(facecolor=c, alpha=0.35, lw=0) for _, c in BANDS] + [Line2D([], [], color=AE, lw=1), Line2D([], [], color=tint, lw=1), Line2D([], [], color="0.3", ls="-", lw=1), Line2D([], [], color="0.3", ls="--", lw=1),
         Line2D([], [], color="k", ls="-.", lw=0.7), Line2D([], [], color="k", ls=(0, (1, 1)), lw=0.9), Line2D([], [], color="0.25", ls=":", lw=0.8)]
    lab = ["early fit", "fitting", "overfitting (N = 1000)", "test probe (NN: test-to-train)", "train probe (NN: train-to-test)", "MNIST", "FM", "gap onset (N = 1000)", "test-recon minimum", "step 100"]
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=6, frameon=False, columnspacing=1.0, handlelength=1.6, labelspacing=0.3)
    for e in ("pdf", "png"): fig.savefig(f"{OUT}/regime_traintest.{e}", dpi=300)
    plt.close(fig)


def end_value(R, m, ds, N, wd, key):
    """per-seed median over snapshots with epoch>500, final snapshot excluded (N=full: last 10 snapshots, ~40 epochs)."""
    out = []
    for r in R.get((m, ds, N, wd), []):
        ep = r["epoch"][:-1]; v = r[key][:-1]; sel = (ep > 500) & np.isfinite(v)
        if not (ep > 500).any():  # N = full only reaches ~42 epochs: use the last 10 snapshots (final excluded) and flag it in the figure
            sel = np.zeros(len(ep), bool); sel[-10:] = True; sel &= np.isfinite(v)
        out.append(np.median(v[sel]) if sel.any() else np.nan)
    return np.array(out)


SHORT = {"gap": "final gap (test/train - 1)", "knn_pix_acc": "kNN pixel acc.", "knn_overlap_pixel": "kNN overlap, pixel", "knn_overlap_prev": "kNN overlap, previous", "knn_overlap_init": "kNN overlap, step 0",
         "rho_err_knn": "Spearman(err, kNN dist)", "test_to_train_nn": "test-to-train NN dist.", "h1_entropy": "H1 entropy", "knn_label_acc": "kNN label acc.", "local_pr_dim": "local PR dim.", "top2_var": "top-2 var. share"}


def acrossN(R):
    items = [("run", "gap")] + [(p, k) for p, k in CUR if p != "topo_test"] + [("topo_test", t) for t in TOPOM]
    nc = 6; nr = int(np.ceil(len(items) / nc)); fig, axs = plt.subplots(nr, nc, figsize=(7.2, 1.45 * nr + 0.5))
    ls_ = {("mnist", "0"): "-", ("fashion_mnist", "0"): "--", ("mnist", "1e-4"): ":", ("fashion_mnist", "1e-4"): (0, (1, 1))}
    for ax, (p, k) in zip(axs.ravel(), items):
        style(ax); ax.set_xscale("log"); ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([250, 1000, 4000, 60000])); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticklabels(["250", "1k", "4k", "60k"]); ax.grid(alpha=0.25, lw=0.4)
        for ds in DS:
            for wd in ("0", "1e-4"):
                xs, mu, sd = [], [], []
                for N in NS:
                    v = end_value(R, "mlp", ds, N, wd, metric_key(p, k)) if (("mlp", ds, N, wd) in R) else np.array([np.nan])
                    if np.isfinite(v).any(): xs.append(NNUM[N]); mu.append(np.nanmean(v)); sd.append(np.nanstd(v))
                if not xs: continue
                xs, mu, sd = map(np.array, (xs, mu, sd))
                ax.errorbar(xs, mu, yerr=sd, color=AE, ls=ls_[(ds, wd)], lw=0.9, marker="o" if ds == "mnist" else "s", ms=2, capsize=1, elinewidth=0.5)
                ax.axvspan(30000, 90000, color="0.9", lw=0, zorder=0)
        ax.set_title(SHORT.get(k, LAB[k]), fontsize=6.3, pad=2)
        if k == "gap": ax.set_yscale("log")
    for ax in axs.ravel()[len(items):]: ax.axis("off")
    for ax in axs[-1]: ax.set_xlabel("N", fontsize=6.5, labelpad=1)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.16, top=0.96, wspace=0.55, hspace=0.7)
    h = [Line2D([], [], color=AE, ls="-"), Line2D([], [], color=AE, ls="--"), Line2D([], [], color=AE, ls=":"), Line2D([], [], color=AE, ls=(0, (1, 1.5)))]
    fig.legend(h, ["MNIST wd 0", "FM wd 0", "MNIST wd 1e-4", "FM wd 1e-4"], loc="lower center", ncol=4,title="end value (epoch > 500; grey band N = 60k: only ~40 epochs, last 10 snapshots)", title_fontsize=6.5, fontsize=7, frameon=False)
    for e in ("pdf", "png"): fig.savefig(f"{OUT}/regime_acrossN.{e}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- table
def near(ep, x): 
    ok = np.where(ep > 0)[0]; return ok[np.argmin(np.abs(np.log(ep[ok]) - np.log(x)))]


def table(R):
    import csv
    mk = json.load(open(f"{TOPO}/regime_numbers.json"))
    rows = []
    items = [("test", g) for g in GEO] + [("cross", c) for c in CROSS] + [("topo_test", t) for t in TOPOM]
    # across-run Spearman (24 MLP AE wd0 runs)
    runs = [(ds, N, s) for ds in DS for N in NS for s in SEEDS]
    gapv = np.array([end_value(R, "mlp", ds, N, "0", "gap")[s] for ds, N, s in runs]); logN = np.log([NNUM[N] for _, N, _ in runs])
    for p, k in items:
        key = metric_key(p, k)
        ev = np.array([end_value(R, "mlp", ds, N, "0", key)[s] for ds, N, s in runs])
        ok = np.isfinite(ev) & np.isfinite(gapv)
        rg = spearmanr(ev[ok], gapv[ok])[0] if ok.sum() > 5 else np.nan; rn = spearmanr(ev[ok], logN[ok])[0] if ok.sum() > 5 else np.nan
        for m in MODELS:
            for ds in DS:
                s = series(R, m, ds, "1000", "0", key)
                if s is None or not np.isfinite(s[1]).any(): continue
                ep, mu, _ = s
                # nearest non-NaN snapshot helper
                def at(x, step0=False):
                    if step0: return mu[0]
                    idx = np.where(np.isfinite(mu) & (ep > 0))[0]; return mu[idx[np.argmin(np.abs(np.log(ep[idx]) - np.log(x)))]]
                tmin = mk[f"1000|{ds}"]["mlp_min"]
                v0, v1, vg, vm = at(0, True), at(1.0), at(G0), at(tmin)
                vend = np.nanmedian(mu[(ep > 500)]) if np.isfinite(mu[ep > 500]).any() else np.nan
                def d(a, b):
                    if not (np.isfinite(a) and np.isfinite(b)): return "NA"
                    rel = (b - a) / max(abs(a), 1e-12); return "↑" if rel > 0.05 else ("↓" if rel < -0.05 else "flat")
                v0d = v0 if np.isfinite(v0) else mu[np.where(np.isfinite(mu))[0][0]]  # direction base: first finite snapshot (overlap_prev/init are NaN at step 0)
                rows.append(dict(metric=k, probe=p, model=m, ds=ds, step0=v0, ep1=v1, gap_onset_33ep=vg, test_min=vm, end=vend,
                                 early_fit=d(v0d, v1), fitting=d(v1, vg), overfitting=d(vg, vend),
                                 spearman_end_vs_gap=rg if m == "mlp" else np.nan, spearman_end_vs_logN=rn if m == "mlp" else np.nan))
    cols = list(rows[0].keys())
    with open(f"{GEOM}/regime_metric_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); [w.writerow(r) for r in rows]
    fm = lambda v: f"{v:.3g}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, (float, np.floating)) else str(v))
    with open(f"{GEOM}/regime_metric_table.md", "w") as f:
        f.write("# Regime metric table (N = 1000, wd 0, test probe, mean over 3 seeds)\n\nValues at step 0, 1 epoch, 33 epochs (MLP AE gap onset), the MLP AE test-recon minimum (119.5 MNIST, 150.8 FM epochs, used for all models), and end (median over epochs > 500, final snapshot excluded). Direction: change between regime endpoints on the mean curve (early fit 0 to 1 epoch, fitting 1 to 33, overfitting 33 to end), threshold 5% of the starting value. Spearman columns are across the 24 MLP AE wd0 runs (2 datasets x 4 N x 3 seeds), end value versus final gap and versus log N.\n\n")
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows: f.write("| " + " | ".join(fm(r[c]) for c in cols) + " |\n")
    gv = spearmanr(gapv, logN)[0]; print("spearman(gap, logN) =", gv)
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    R = load_all(); print("loaded", len(R), "groups")
    if what in ("full", "all"): full_fig(R)
    if what in ("curated", "all"): curated(R)
    if what in ("traintest", "all"): traintest(R)
    if what in ("acrossN", "all"): acrossN(R)
    if what in ("table", "all"): table(R)
