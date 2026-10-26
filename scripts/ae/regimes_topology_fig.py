"""Regimes figure: affine share R1 + three scale-free topological summaries vs epoch, classifier vs AE (CPU only).
Reads: classifier topo cache (analysis_v2/cache), AE topo cache (ae_dynamics/analysis/topo; scripts/ae/ae_topo_cache.py), R1 via latentdyn.ae.lds_load.
Writes: dynamics_report/figures/onepage_regimes.{pdf,png}, regimes_topology_full.png, ae_dynamics/analysis/topo/regime_numbers.json"""
from latentdyn._threads import cap_threads; cap_threads(4)
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from latentdyn.ae import lds_load as L
L.clf_gen = lambda rid, probe: None
BV = "/vol/bitbucket/lbastian/latent-dynamics"; CACHE = f"{BV}/analysis_v2/cache"; TOPO = f"{BV}/ae_dynamics/analysis/topo"; OUT = f"{BV}/dynamics_report/figures"
CL, AE, LIN = "#d62728", "#1f77b4", "#7f7f7f"
DS = ("mnist", "fashion_mnist"); SEEDS = (0, 1, 2); MODELS = ("clf", "mlp", "lin")
COL = {"clf": CL, "mlp": AE, "lin": LIN}; NAME = {"clf": "classifier", "mlp": "MLP AE", "lin": "linear AE"}
LS = {"mnist": "-", "fashion_mnist": "--"}; DL = {"mnist": "MNIST", "fashion_mnist": "FM"}
METRICS = [("h1_entropy", "H1 entropy"), ("h1_total", "H1 lifetime sum"), ("pmag_fixed", "PMag")]


def run_id(m, ds, s, N):
    return f"{ds}_probe_none_d16_s{s}_phase_n{N}_wd0" if m == "clf" else f"{ds}_ae_{'mlp' if m == 'mlp' else 'linear'}_d16_s{s}_n{N}_wd0"


def load_topo(m, ds, s, N, probe="test"):
    f = f"{CACHE}/{run_id(m, ds, s, N)}.npz" if m == "clf" else f"{TOPO}/{run_id(m, ds, s, N)}.npz"
    if not os.path.isfile(f): return None
    z = dict(np.load(f)); r = {"step": z["step"], "epoch": z["run/epoch"], "mask": z["topo_mask"]}
    for k in ("train_acc", "test_acc", "recon_train", "recon_test"):
        if "run/" + k in z: r[k] = z["run/" + k]
    for k, _ in METRICS: r[k] = z[f"test/{k}"]
    ft = f.replace(".npz", "__train.npz") if m != "clf" else f
    if probe == "train" or True:
        if m == "clf": zt = z
        elif os.path.isfile(ft): zt = dict(np.load(ft))
        else: zt = None
        if zt is not None and "train/h1_entropy" in zt:
            for k, _ in METRICS: r["tr_" + k] = zt[f"train/{k}"]
    return r


def load_all():
    T = {}
    for N in ("1000", "full"):
        for ds in DS:
            for m in MODELS:
                if N == "full" and False: continue
                T[(N, ds, m)] = [load_topo(m, ds, s, N) for s in SEEDS]
    R1 = {}
    for ds in DS:
        for r in L.load_classifier(ds, "test"):
            if r["wd"] == 0.0: R1[("clf", ds, str(r["N"]) if r["N"] != 60000 else "full", r["seed"])] = (r["steps"][1:], r["_P"][0])
    for r in L.load_ae("lab"):
        if r["probe"] == "test" and r["d"] == 16 and r["wd"] == 0.0:
            R1[(r["arch"] if r["arch"] == "linear" else r["arch"], r["dataset"], str(r["N"]) if r["N"] != 60000 else "full", r["seed"])] = (r["steps"][1:], r["_P"][0])
    R1 = {(("lin" if k[0] == "linear" else k[0]),) + k[1:]: v for k, v in R1.items()}
    return T, R1


def agg(arrs):
    A = np.array(arrs, float); return np.nanmean(A, 0), np.nanstd(A, 0)


def markers(T):
    """epochs (mean over 3 seeds) per (N, ds): AE gap onset & test min (MLP and linear), classifier saturation & test-acc max."""
    M = {}
    for N in ("1000", "full"):
        for ds in DS:
            d = {}
            for m in ("mlp", "lin"):
                g, mn = [], []
                for r in T[(N, ds, m)]:
                    if r is None: continue
                    rt, rr, ep = r["recon_test"], r["recon_train"], r["epoch"]
                    k = np.where(rt / rr - 1 > 0.10)[0]; g.append(ep[k[0]] if len(k) else np.nan); mn.append(ep[np.nanargmin(rt)])
                d[m + "_gap"] = float(np.nanmean(g)) if len(g) else np.nan; d[m + "_min"] = float(np.nanmean(mn)) if len(mn) else np.nan
                d[m + "_gap_seeds"] = [float(x) for x in g]; d[m + "_min_seeds"] = [float(x) for x in mn]
            sat, mx = [], []
            for r in T[(N, ds, "clf")]:
                k = np.where(r["train_acc"] >= 0.999)[0]; sat.append(r["epoch"][k[0]] if len(k) else np.nan); mx.append(r["epoch"][np.nanargmax(r["test_acc"])])
            d["clf_sat"] = float(np.nanmean(sat)); d["clf_max"] = float(np.nanmean(mx)); d["clf_sat_seeds"] = [float(x) for x in sat]; d["clf_max_seeds"] = [float(x) for x in mx]
            M[(N, ds)] = d
    return M


def series(T, R1, N, ds, m, key):
    """returns (epoch, mean, sd) for metric key ('R1' or topo metric, prefix 'tr_' for train) for model m."""
    runs = T[(N, ds, m)]
    if key == "R1":
        items = [R1.get((m, ds, N, s)) for s in SEEDS]
        if any(i is None for i in items): return None
        st = items[0][0]; assert all(np.array_equal(i[0], st) for i in items)
        ratio = [r["epoch"][-1] / r["step"][-1] for r in runs]; ep = st * np.mean(ratio)
        mu, sd = agg([i[1] for i in items]); return ep, mu, sd
    if any(r is None or key not in r for r in runs): return None
    mu, sd = agg([r[key] for r in runs]); return np.mean([r["epoch"] for r in runs], 0), mu, sd


def style(ax, ylabel_only=False):
    ax.set_xscale("log"); ax.grid(alpha=0.25, lw=0.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ax.spines.values(): s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2, pad=1.5)


def draw(ax, T, R1, N, key, M, lw=0.9, band=True):
    for ds in DS:
        for m in MODELS:
            s = series(T, R1, N, ds, m, key)
            if s is None: continue
            ep, mu, sd = s; ok = np.isfinite(mu) & (ep > 0)
            ax.plot(ep[ok], mu[ok], color=COL[m], ls=LS[ds], lw=lw)
            if band: ax.fill_between(ep[ok], (mu - sd)[ok], (mu + sd)[ok], color=COL[m], alpha=0.15, lw=0)


def vlines(ax, M, N, lw=0.6):
    e = lambda k: float(np.nanmean([M[(N, d)][k] for d in DS]))
    for x, c in ((1, "k"), (100 * 0.125 if N == "1000" else 100 * 2500 / 20000 * 0 + 100 * (1.0 / 469), "k"), (e("mlp_gap"), AE), (e("mlp_min"), AE), (e("clf_sat"), CL)):
        if x is not None and np.isfinite(x): ax.axvline(x, color=c, ls=":", lw=lw, alpha=0.9, zorder=0)


def one_page(T, R1, M):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.75))
    keys = [("R1", "affine share R1")] + [(k, t) for k, t in (("h1_entropy", "H1 entropy"), ("h1_total", "H1 lifetime sum"), ("pmag_fixed", "PMag"))]
    N = "1000"
    for ax, (k, t) in zip(axs, keys):
        style(ax); draw(ax, T, R1, N, k, M); vlines(ax, M, N); ax.set_title(t, fontsize=7, pad=8 if k == "R1" else 2); ax.set_xlim(0.125, 2560); ax.set_xlabel("epoch", fontsize=7, labelpad=1)
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6)); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # regime labels, panel (a)
    a = axs[0]; gap = float(np.nanmean([M[(N, d)]["mlp_gap"] for d in DS]))
    tr = matplotlib.transforms.blended_transform_factory(a.transData, a.transAxes)
    for x, t in ((np.sqrt(0.125 * 1), "early fit"), (np.sqrt(1 * gap), "fitting"), (np.sqrt(gap * 2560), "overfitting")):
        a.text(x, 1.02, t, transform=tr, ha="center", va="bottom", fontsize=5.5, color="0.25")
    h = [Line2D([], [], color=COL[m], lw=1) for m in MODELS] + [Line2D([], [], color="0.3", ls=LS[d], lw=1) for d in DS]
    axs[int(os.environ.get("LEG_PANEL", 3))].legend(h, [NAME[m] for m in MODELS] + ["MNIST", "FM"], fontsize=5.5, frameon=False, loc=os.environ.get("LEG_LOC", "best"), handlelength=1.6, labelspacing=0.2, borderpad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    os.makedirs(OUT, exist_ok=True); fig.savefig(f"{OUT}/onepage_regimes.pdf"); fig.savefig(f"{OUT}/onepage_regimes.png", dpi=300); plt.close(fig)


def full_fig(T, R1, M):
    have_train = all(r is not None and "tr_h1_entropy" in r for m in ("mlp", "lin") for r in T[("1000", "mnist", m)])
    nrow = 2 if have_train else 1
    fig, axs = plt.subplots(nrow * 2, 4, figsize=(12, 3.0 * nrow * 2 * 0.85), squeeze=False)
    keys = [("R1", "affine share R1")] + METRICS
    for i, N in enumerate(("1000", "full")):
        for ax, (k, t) in zip(axs[i], keys):
            style(ax); draw(ax, T, R1, N, k, M, lw=1.0); vlines(ax, M, N); ax.set_title(f"{t}, N={'1000' if N=='1000' else '60000'}", fontsize=8); ax.set_xlabel("epoch", fontsize=7)
    if have_train:
        for i, N in enumerate(("1000", "full")):
            for ax, (k, t) in zip(axs[2 + i], [(None, None)] + METRICS):
                if k is None: ax.axis("off"); continue
                style(ax)
                for ds in DS:
                    for m in MODELS:
                        a, b = series(T, R1, N, ds, m, k), series(T, R1, N, ds, m, "tr_" + k)
                        if a is None or b is None: continue
                        ok = np.isfinite(a[1]) & np.isfinite(b[1]) & (a[0] > 0)
                        ax.plot(a[0][ok], (b[1] - a[1])[ok], color=COL[m], ls=LS[ds], lw=1.0)
                ax.axhline(0, color="k", lw=0.4); vlines(ax, M, N); ax.set_title(f"train - test: {t}, N={'1000' if N=='1000' else '60000'}", fontsize=8); ax.set_xlabel("epoch", fontsize=7)
    h = [Line2D([], [], color=COL[m], lw=1) for m in MODELS] + [Line2D([], [], color="0.3", ls=LS[d], lw=1) for d in DS]
    axs[0, 0].legend(h, [NAME[m] for m in MODELS] + ["MNIST", "FM"], fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/regimes_topology_full.png", dpi=130); plt.close(fig)
    return have_train


if __name__ == "__main__":
    T, R1 = load_all(); M = markers(T)
    one_page(T, R1, M); ht = full_fig(T, R1, M)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in M.items()}, open(f"{TOPO}/regime_numbers.json", "w"), indent=1)
    for k, v in M.items(): print(k, {a: round(b, 1) for a, b in v.items() if not a.endswith("seeds")})
    print("train rows:", ht)
