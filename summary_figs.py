"""Six summary figures for the latent-dynamics headline results (ICML column widths).
Usage: python summary_figs.py [fig1 fig2 ...]   (default: all).  Reads existing analysis outputs only,
except the shuffled-label null for fig4, which is computed once (cheap, CPU) and cached next to the figures."""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import csv, glob, json, re, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

V2 = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2"
OUT = f"{V2}/summary_figs"
os.makedirs(OUT, exist_ok=True)
W1, W2 = 3.3, 6.8
# Okabe-Ito colour-blind-safe palette
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", sky="#56B4E9", pink="#CC79A7", yellow="#F0E442", black="#000000", grey="#777777")
SEQ = [OI["blue"], OI["sky"], OI["green"], OI["orange"], OI["verm"], OI["pink"]]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "legend.frameon": False,
                     "figure.dpi": 100, "pdf.fonttype": 42, "font.family": "DejaVu Sans", "xtick.major.width": 0.7, "ytick.major.width": 0.7})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", name)


def rd(path): return list(csv.DictReader(open(path)))
def tag(ax, s, x=-0.02, y=1.04): ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="right")


# ---------------------------------------------------------------- fig1
def fig1():
    rows = [r for r in rd(f"{V2}/vfield/decomp/summary_runs.csv") if r["family"] == "phase" and r["probe"] == "test"]; assert len(rows) == 75
    bands = [("b1_10", "1–10"), ("b10_100", "10–100"), ("b100_1000", "100–1k"), ("b1000_up", "$\\geq$1k")]
    med = lambda k: float(np.nanmedian([float(r[k]) if r[k] != "" else np.nan for r in rows]))
    tot = np.array([[med(f"R1_affine_{b}"), med(f"R2_classaffine_{b}") - med(f"R1_affine_{b}"), med(f"R3_krr_{b}") - med(f"R2_classaffine_{b}"), 1 - med(f"R3_krr_{b}")] for b, _ in bands])
    res = np.array([[med(f"resid_frac_classaffine_{b}"), med(f"resid_frac_nonlin_krr_{b}")] for b, _ in bands]); res = np.c_[res, 1 - res.sum(1)]
    print("fig1 totals", np.round(tot, 3).tolist(), "resid", np.round(res, 3).tolist())
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(wspace=0.32)); x = np.arange(4); lab = [l for _, l in bands]
    cols = [OI["blue"], OI["orange"], OI["green"], "#BBBBBB"]
    names = ["shared linear map", "+ per-class linear", "+ nonlinear", "unexplained"]
    bot = np.zeros(4)
    for j in range(4):
        axs[0].bar(x, tot[:, j], 0.7, bottom=bot, color=cols[j], label=names[j], edgecolor="white", linewidth=0.4); bot += tot[:, j]
    axs[0].set_ylim(0, 1); axs[0].set_ylabel("fraction of held-out motion"); axs[0].set_xlabel("training step"); axs[0].set_xticks(x, lab)
    axs[0].set_title("total motion", loc="left"); axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2, columnspacing=1, handlelength=1.2, fontsize=6.8)
    for i in range(4): axs[0].text(x[i], tot[i, 0] / 2, f"{100*tot[i,0]:.0f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    bot = np.zeros(4); rc = [cols[1], cols[2], cols[3]]
    for j in range(3):
        axs[1].bar(x, res[:, j], 0.7, bottom=bot, color=rc[j], edgecolor="white", linewidth=0.4); bot += res[:, j]
    for i in range(4): axs[1].text(x[i], res[i, :2].sum() + res[i, 2] / 2, f"{100*res[i,2]:.0f}%", ha="center", va="center", fontsize=7, fontweight="bold")
    axs[1].set_ylim(0, 1); axs[1].set_ylabel("fraction of post-linear residual"); axs[1].set_xlabel("training step"); axs[1].set_xticks(x, lab)
    axs[1].set_title("after removing the shared linear map", loc="left")
    tag(axs[0], "a", -0.18); tag(axs[1], "b", -0.18)
    save(fig, "fig1_partition")


# ---------------------------------------------------------------- fig2
TAGS = [("dl_adam_n1000_wd0", "Adam N=1000", OI["blue"], "-"), ("dl_adam_n4000_wd0", "Adam N=4000", OI["sky"], "-"), ("dl_adam_nfull_wd0", "Adam N=60000", OI["green"], "-"),
        ("dl_adam_n4000_wd1e-3", "Adam N=4000, wd 1e-3", OI["orange"], "-"), ("dl_adam_n4000_ln0.2", "Adam N=4000, 20% noise", OI["pink"], "-"), ("dl_sgd_lr0.025_n4000_wd0", "SGD N=4000", OI["verm"], "--")]


def fig2():
    A = f"{V2}/dense_latent/analysis"; fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(wspace=0.35)); allv = []
    for tg, lab, c, ls in TAGS:
        rows = [r for r in rd(f"{A}/q13_lag_{tg}.csv") if r["probe"] == "test" and r["frame"] == "latent" and r["snap_equiv"] == "0"]
        t = np.array([float(r["t"]) for r in rows]); lag = np.array([float(r["lag"]) for r in rows]); Ea = np.array([float(r["E_res_aff"]) for r in rows]); Ek = np.array([float(r["E_res_krr"]) for r in rows])
        Fl = np.array([float(r["E_floor"]) for r in rows]); clean = (Fl / Ek < 0.1)                # fp16 rounding floor negligible
        ok = (t >= 48) & (lag <= np.maximum(t, 1)) & (lag <= 1000) & clean; L = np.unique(lag[ok])
        cnt = {l: int((ok & (lag == l)).sum()) for l in L}; L = np.array([l for l in L if cnt[l] >= 2])
        share = [np.median(Ek[ok & (lag == l)] / Ea[ok & (lag == l)]) for l in L]
        cb = [b for b in np.unique(t[t >= 4]) if clean[(t == b) & (lag == 1)].all()]; L2 = []; en = []
        for l in np.unique(lag):
            v = [Ek[(t == b) & (lag == l)][0] / Ek[(t == b) & (lag == 1)][0] for b in cb if ((t == b) & (lag == l)).any() and clean[(t == b) & (lag == l)][0] and l <= max(b, 1)]
            if len(v) >= 2: L2.append(l); en.append(np.median(v))
        axs[0].plot(L, share, color=c, ls=ls, label=lab, marker="o", ms=2); allv += share
        axs[1].plot(L2, np.array(en), color=c, ls=ls, marker="o", ms=2)
    axs[0].set_xscale("log"); axs[0].set_ylim(0, 1.05); axs[0].set_xlabel("lag $\\Delta$ (steps)"); axs[0].set_ylabel("unexplained / post-linear residual")
    axs[0].axhspan(0.7, 0.9, color="0.9", zorder=0); axs[0].set_title("unexplained share vs lag", loc="left"); axs[0].set_xlim(0.8, 1500)
    axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("lag $\\Delta$ (steps)"); axs[1].set_ylabel("residual energy / value at $\\Delta$=1")
    x = np.array([1., 1000.]); axs[1].plot(x, x, color="k", lw=0.8, ls=":"); axs[1].plot(x, x ** 2, color="k", lw=0.8, ls="-.")
    axs[1].set_ylim(0.5, 2e6); axs[1].set_xlim(0.8, 1500); axs[1].set_title("residual energy vs lag", loc="left")
    h = [Line2D([], [], color=c, ls=ls, label=lab) for _, lab, c, ls in TAGS] + [Line2D([], [], color="k", lw=0.8, ls=":", label="slope 1 (random walk)"), Line2D([], [], color="k", lw=0.8, ls="-.", label="slope 2 (drift)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.26), fontsize=7, columnspacing=1.2, handlelength=1.8)
    tag(axs[0], "a", -0.17); tag(axs[1], "b", -0.17); print("fig2 share range", min(allv), max(allv), np.median(allv))
    save(fig, "fig2_lag")


# ---------------------------------------------------------------- fig3
def fig3():
    K = f"{V2}/ntk_coupling"; fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(wspace=0.32)); endr = {}
    for tg, lab, c, ls in TAGS:
        fs = sorted(glob.glob(f"{K}/mnist_probe_none_d16_s0_dl_{tg[3:]}/pair_*.json")); rs = sorted((json.load(open(f)) for f in fs), key=lambda r: r["step"])
        st = np.array([r["step"] for r in rs]); ck = lambda k: np.array([r["c_kernel"].get("pre", r["c_kernel"].get("raw")).get(k, np.nan) for r in rs], float)
        axs[0].plot(st, ck("fro_ratio_wb"), color=c, ls=ls, label=lab); endr[tg] = (np.nanmedian(ck("fro_ratio_wb")), np.nanmedian(ck("cosn_within")), np.nanmedian(ck("cosn_between")))
        if tg == "dl_adam_n4000_wd0":
            axs[1].plot(st, ck("cosn_within"), color=OI["blue"], label="within class"); axs[1].plot(st, ck("cosn_between"), color=OI["verm"], ls="--", label="between classes")
    axs[0].axhline(1, color="k", lw=0.6, ls=":"); axs[0].set_xscale("log"); axs[0].set_xlabel("training step"); axs[0].set_ylabel("$\\|K\\|_F$ within / between class")
    axs[0].set_title("all runs", loc="left"); axs[1].set_xscale("log"); axs[1].set_xlabel("training step"); axs[1].set_ylabel("cosine-normalised coupling")
    axs[1].set_title("Adam, N=4000", loc="left"); axs[1].legend(loc="best")
    h = [Line2D([], [], color=c, ls=ls, label=lab) for _, lab, c, ls in TAGS]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.27, -0.2), fontsize=7, columnspacing=1.2, handlelength=1.8)
    tag(axs[0], "a", -0.17); tag(axs[1], "b", -0.17); print("fig3 medians (fro, cosn_w, cosn_b)", {k: np.round(v, 3).tolist() for k, v in endr.items()})
    save(fig, "fig3_kernel")


# ---------------------------------------------------------------- fig4
def spectra_null(N, seed=0):
    f = f"{OUT}/spectra_null_cache_n{N}_s{seed}.npz"
    if os.path.exists(f): return np.load(f)["steps"], np.load(f)["absc"]
    import torch; torch.set_num_threads(8); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vfield as Vf, vfield_spectra as S
    rid = f"mnist_probe_none_d16_s{seed}_phase_n{N if N != 60000 else 'full'}_wd0"; Z, steps, y, noisy = Vf.load_run(f"{Vf.RUNS}/{rid}", "test"); ds = np.diff(steps).astype(float)
    rng = np.random.RandomState(0); yp = rng.permutation(y); classes = np.unique(yp); tr, te = Vf.make_split(yp, 0)
    Zt = torch.tensor(Z, dtype=torch.float64); Ag, R = S.class_fits(Zt, yp, classes, tr, torch.device("cpu")); em = S.eig_metrics(R / ds[:, None, None, None])
    np.savez(f, steps=steps[1:], absc=em["abscissa_cls"]); return steps[1:], em["abscissa_cls"]


def fig4():
    sp = f"{V2}/vfield/spectra/runs"; Ns = [250, 1000, 4000, 16000, 60000]; cm = dict(zip(Ns, [OI["sky"], OI["blue"], OI["green"], OI["orange"], OI["verm"]]))
    fig, ax = plt.subplots(figsize=(W1, 2.6)); mins = []; nullmax = []; nulls = []
    for N in Ns:
        fs = sorted(glob.glob(f"{sp}/mnist_probe_none_d16_s*_phase_n{N if N!=60000 else 'full'}_wd0_test_given.npz")) or sorted(glob.glob(f"{sp}/mnist_probe_none_d16_s*_phase_n{N}_wd0_test_given.npz"))
        assert len(fs) == 3, (N, fs)
        zs = [np.load(f) for f in fs]; st = zs[0]["steps"][1:]; arr = np.concatenate([z["abscissa_cls_res"] for z in zs], 1)     # (119, 3*10)
        m = np.nanmedian(arr, 1); lo, hi = np.nanpercentile(arr, [25, 75], 1); mins.append(np.nanmin(arr))
        ax.plot(st, m, color=cm[N], label=f"N={N:,}"); ax.fill_between(st, lo, hi, color=cm[N], alpha=0.12, lw=0)
        ns, na = spectra_null(N); nulls.append((ns, np.nanmedian(na, 1)))
    L = min(len(n[0]) for n in nulls); nl = np.array([n[1][:L] for n in nulls]); ax.fill_between(nulls[0][0][:L], nl.min(0), nl.max(0), color="0.6", alpha=0.45, lw=0, zorder=0); nullmax = [float(nl.max())]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("leading real eigenvalue of\nper-class residual generator (step$^{-1}$)")
    h, l = ax.get_legend_handles_labels(); h.append(plt.Rectangle((0, 0), 1, 1, color="0.6", alpha=0.45, lw=0)); l.append("shuffled-label null\n(range over N)")
    ax.legend(h, l, loc="lower left", fontsize=6.8); print("fig4 min over all per-class series:", min(mins), "null max", nullmax)
    save(fig, "fig4_spectra")


# ---------------------------------------------------------------- fig5
def fig5():
    D = f"{V2}/vfield/decomp"; summ = [r for r in rd(f"{D}/summary_runs.csv") if r["family"] == "gap" and r["probe"] == "train" and r["ln"] == "0.2"]
    Ns = [250, 1000, 4000, 16000, 60000]; cm = dict(zip(Ns, [OI["sky"], OI["blue"], OI["green"], OI["orange"], OI["verm"]])); fig, ax = plt.subplots(figsize=(W1, 2.6)); info = {}; lines_ = {}
    for N in Ns:
        curves = []
        for s in (0, 1, 2):
            z = np.load(f"{D}/runs/mnist_probe_none_d16_s{s}_gap_n{N if N!=60000 else 'full'}_ln0.2_train.npz", allow_pickle=True)
            curves.append(z["nl_mis_f1_cos_given"] - z["nl_mis_f1_cos_true"]); st = z["steps"][1:]
        d = np.median(curves, 0); ep = st * 128.0 / N; ok = ep > 0
        ln_, = ax.plot(ep[ok], d[ok], color=cm[N]); lines_[N] = ln_
        cr = np.nan
        for k in range(len(d)):
            if d[k] > 0 and np.mean(d[k:] > 0) >= 0.8: cr = ep[k]; break    # same rule as vfield_decomp_aggregate.pcross, applied to the plotted median curve
        if np.isfinite(cr): ax.plot([cr], [np.interp(cr, ep[ok], d[ok])], "o", ms=5.5, mfc="white", mec=cm[N], mew=1.6, zorder=5)
        info[N] = round(float(cr), 2)
    ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log"); ax.set_xlim(1e-2, 1e3); ax.set_xlabel("epoch"); ax.set_ylabel("cos(Δz, → given-class mean)\n− cos(Δz, → true-class mean)")
    mem = {}
    for N in Ns:
        v = [float(r["perclass_mem_cross_epoch"]) for r in rd(f"{V2}/vfield/summary_runs.csv") if r["family"] == "gap" and r["probe"] == "train" and r["ln"] == "0.2" and int(r["N"]) == N and r["perclass_mem_cross_epoch"] not in ("", "nan")]
        mem[N] = float(np.median(v)); ax.plot([mem[N]], [0], marker="^", ms=6.5, color=cm[N], mec="k", mew=0.5, zorder=6, clip_on=False)
    h = [Line2D([], [], color=cm[N], lw=1.3) for N in Ns] + [Line2D([], [], marker="o", ls="", mfc="white", mec="k", mew=1.4, ms=5), Line2D([], [], marker="^", ls="", color="0.5", mec="k", mew=0.5, ms=6.5)]
    l = [f"N={N:,}" for N in Ns] + ["field flip (curve crosses 0)", "memorisation crossover (NCM)"]
    ax.legend(h, l, loc="upper left", fontsize=6.5); print("fig5 memorisation crossover epochs", {k: round(v, 1) for k, v in mem.items()}); print("fig5 crossover epochs", info)
    save(fig, "fig5_mislabeled")


# ---------------------------------------------------------------- fig6
def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(wspace=0.32, width_ratios=[1.25, 1]))
    rid = "mnist_probe_none_d16_s0_eps1e-8_n4000_wd1e-4"; Dd = f"{V2}/dense/runs/{rid}"; a = np.load(f"{Dd}/dense_scalars.npy"); cols = json.load(open(f"{Dd}/dense_scalars_cols.json")); loss = a[:, cols.index("loss")]; step = a[:, cols.index("step")]
    snaps = np.array(sorted(int(re.search(r"step_(\d+)", f).group(1)) for f in os.listdir(f"{Dd}/latents"))); snaps = snaps[snaps > 0]
    ax = axs[0]; ax.plot(step, loss, color=OI["blue"], lw=0.4, rasterized=True, label="per-step loss")
    sel = np.isin(step, snaps); ax.plot(step[sel & (step >= 1000)], loss[sel & (step >= 1000)], color=OI["verm"], marker="o", ms=3, lw=1.1, label="loss seen at snapshots", zorder=5)
    ax.set_yscale("log"); ax.set_xlim(1000, 40000); ax.set_ylim(loss[step >= 1000].min() * 0.5, 3)
    base = np.array([np.median(loss[max(0, i - 100):i + 100]) for i in range(len(loss))]); spike = np.where(loss > 10 * base)[0] + 1
    spike = spike[spike > 1000]; groups = np.split(spike, np.where(np.diff(spike) > 5)[0] + 1) if len(spike) else []; peaks = [g[np.argmax(loss[g - 1])] for g in groups if len(g)]
    near = lambda s_: np.min(np.abs(snaps - s_)) <= 2; nnear = sum(near(p) for p in peaks)
    ax.set_xlabel("training step"); ax.set_ylabel("minibatch train loss"); ax.set_title("Adam, N=4000, wd=1e-4", loc="left"); ax.legend(loc="upper right", fontsize=6.8)
    print("fig6a spikes (>10x local median, step>1000):", len(peaks), "within 2 steps of a snapshot:", nnear)
    # (b)
    P = [r for r in rd(f"{V2}/sgd_lrB/points.csv") if r["N"] == "4000" and r["rule"] == "t" and float(r["level"]) == 240.0 and r["metric"] == "fisher" and r["value"] not in ("", "nan")]
    ax = axs[1]; out = {}; from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    pts = [r for r in P if r["line"] == "lr_fixed"]; g = collections.defaultdict(list)
    for r in pts: g[float(r["nu"])].append(float(r["value"]))
    nus = sorted(g); m = np.array([np.mean(g[n]) for n in nus]); e = np.array([np.std(g[n]) for n in nus])
    ax.errorbar(nus, m, yerr=e, color=OI["verm"], marker="o", ms=4, capsize=2, lw=1.2, label="lr = 0.025 fixed, B = 32...512"); out["lr_fixed"] = (nus, np.round(m, 3).tolist())
    g2 = collections.defaultdict(list); nu0 = None
    for r in P:
        if r["line"] == "nu_fixed": g2[float(r["lr"])].append(float(r["value"])); nu0 = float(r["nu"])
    lrs = sorted(g2); xs = nu0 * np.exp(np.linspace(-0.03, 0.03, len(lrs))); mm = np.array([np.mean(g2[l]) for l in lrs]); ee = np.array([np.std(g2[l]) for l in lrs])
    ax.errorbar(xs, mm, yerr=ee, color=OI["blue"], marker="s", ms=4, capsize=2, lw=0, elinewidth=1.2, label="$\\nu$ = 1.95e-4 fixed, lr = 0.006...0.1"); out["nu_fixed"] = (nu0, np.round(mm, 3).tolist()); ax.annotate("lr 0.006→0.1\nat fixed $\\nu$", (nu0, mm.max() + ee.max()), xytext=(nu0 * 1.0, 4.15), ha="center", fontsize=6.8, color=OI["blue"], arrowprops=dict(arrowstyle="-", color=OI["blue"], lw=0.6))
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator([5e-5, 1e-4, 2e-4, 4e-4, 8e-4])); ax.xaxis.set_major_formatter(FixedFormatter(["5e-5", "1e-4", "2e-4", "4e-4", "8e-4"])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("noise scale $\\nu$ = lr / B"); ax.set_ylabel("Fisher separation at $t$=240"); ax.set_title("SGD, N=4000", loc="left"); ax.legend(loc="upper left", fontsize=6.3); ax.set_ylim(2.1, 5.3)
    print("fig6b", out); tag(axs[0], "a", -0.15); tag(axs[1], "b", -0.2)
    save(fig, "fig6_spikes")


# ---------------------------------------------------------------- fig7
def fig7():
    rows = rd(f"{V2}/vfield/traintest/summary_runs.csv"); fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(wspace=0.34))
    ax = axs[0]; Ns = [250, 1000, 4000, 16000, 60000]; x = np.arange(len(Ns)); info = {}
    for key, lab, c, mk, off in [("inc_class_sep_within_te_b1000_up", "fit on test (within)", OI["blue"], "o", -0.09), ("inc_class_sep_cross_tr_to_te_b1000_up", "fit on train (cross)", OI["orange"], "s", 0.09)]:
        m = []; lo = []; hi = []
        for N in Ns:
            v = np.array([float(r[key]) for r in rows if r["family"] == "phase" and r["wd"] == "0.0" and int(r["N"]) == N]); m.append(np.median(v)); lo.append(v.min()); hi.append(v.max())
        m, lo, hi = map(np.array, (m, lo, hi)); ax.errorbar(x + off, m, yerr=[m - lo, hi - m], color=c, marker=mk, ms=4.5, capsize=2, lw=0, elinewidth=1.1, label=lab); info[lab] = np.round(m, 4).tolist()
    ax.set_xticks(x, [f"{N:,}" for N in Ns], fontsize=7.5); ax.set_xlabel("training-set size N (wd = 0)"); ax.set_ylabel("class-wise $R^2$ gain, test points\n(steps $\\geq$1000)"); ax.set_yscale("symlog", linthresh=2e-3); ax.set_ylim(-4e-3, 0.15); ax.axhline(0, color="k", lw=0.5); ax.set_yticks([-2e-3, 0, 2e-3, 1e-2, 1e-1], ["-0.002", "0", "0.002", "0.01", "0.1"])
    ax.set_title("same field", loc="left"); ax.legend(loc="upper left", fontsize=6.8); print("fig7a median class-wise increments", info)
    ax = axs[1]; wds = sorted({float(r["wd"]) for r in rows}); nz = [w for w in wds if w > 0]; cv = plt.get_cmap("viridis"); cmap = {w: cv(0.95 - 0.7 * (np.log10(w) - np.log10(min(nz))) / (np.log10(max(nz)) - np.log10(min(nz)))) for w in nz}; cmap[0.0] = "#000000"
    k = "trratio_log_tr_over_te_b1000_up"; X = []; Y = []; Xn = []; Yn = []
    for r in rows:
        y = float(r[k]); xx = 1.0 / float(r["N"]); X.append(xx); Y.append(y)
        noisy = r["family"] == "gap"; ax.scatter(xx, y, s=16 if not noisy else 22, marker="^" if noisy else "o", color=cmap[float(r["wd"])], alpha=0.6, lw=0, zorder=3)
    X = np.array(X); Y = np.array(Y); ok = np.isfinite(Y); rr = np.corrcoef(X[ok], Y[ok])[0, 1]
    ns = np.array([r["ln"] == "0.0" for r in rows]); rr0 = np.corrcoef(X[ok & ns], Y[ok & ns])[0, 1]
    ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log"); ax.set_xlabel("1 / N"); ax.set_ylabel("log(train / test class spread)\n(<0: train more collapsed)")
    ax.set_title("different geometry", loc="left")
    h = [Line2D([], [], marker="o", ls="", color=cmap[w], ms=4.5, label=("wd=0" if w == 0 else f"wd={w:g}")) for w in wds] + [Line2D([], [], marker="^", ls="", color="0.4", ms=5, label="label noise")]
    ax.legend(handles=h, loc="lower left", fontsize=6.3, ncol=3, columnspacing=0.7, handletextpad=0.2, title=f"Pearson r with 1/N = {rr:.2f}", title_fontsize=6.8)
    print("fig7b Pearson r all", rr, "no label noise", rr0, "n", int(ok.sum()))
    tag(axs[0], "a", -0.24); tag(axs[1], "b", -0.2); save(fig, "fig7_traintest")


if __name__ == "__main__":
    allf = dict(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6, fig7=fig7)
    for k in (sys.argv[1:] or allf): allf[k]()
