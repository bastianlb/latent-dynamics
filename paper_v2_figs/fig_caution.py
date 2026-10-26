"""Methods-caution figure: (a) PMag window, (b) l2 temperature, (c) H1 total vs Fisher redundancy."""
from common import *
from scipy.stats import spearmanr

setup()
fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.15), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1.05, 1]))

# (a) PMag: per-snapshot window vs dataset-shared window, 27 MNIST runs (v1 grid, reanalysis_v2)
rows = [r for r in read_csv(f"{RE2}/pmag_windows_final.csv") if r["dataset"] == "mnist"]
x = np.array([fnum(r["pmag_csv"]) for r in rows])
y = np.array([fnum(r["B"]) for r in rows])
rho = spearmanr(x, y).correlation
rho_f, rho_c10 = [spearmanr([fnum(r["pmag_csv"]) for r in read_csv(f"{RE2}/pmag_windows_final.csv") if r["dataset"] == d], [fnum(r["B"]) for r in read_csv(f"{RE2}/pmag_windows_final.csv") if r["dataset"] == d]).correlation for d in ("fashion_mnist", "cifar10")]
ax = axes[0]
cmap = {"none": C["blue"], "l2": C["verm"], "bn": C["green"]}
mk = {"probe": "o", "probe_recon": "s", "autoencoder": "^"}
for nm in ["none", "l2", "bn"]:
    for ob in ["probe", "probe_recon", "autoencoder"]:
        s = [i for i, r in enumerate(rows) if r["norm"] == nm and r["objective"] == ob]
        ax.scatter(x[s], y[s], color=cmap[nm], marker=mk[ob], s=11, lw=0, alpha=0.9)
ax.set_xlabel("PMag, per-snapshot window")
ax.set_ylabel("PMag, shared fixed window")
ax.text(0.04, 0.05, "Spearman $\\rho$:\n" + rf"MNIST ${rho:+.2f}$" + "\n" + rf"Fashion ${rho_f:+.2f}$" + "\n" + rf"CIFAR-10 ${rho_c10:+.2f}$", transform=ax.transAxes, fontsize=5.8, va="bottom")
h = [plt.Line2D([], [], marker="o", ls="", color=cmap[k], ms=3.5, label=lab)
     for k, lab in [("none", "none"), ("l2", r"$\ell_2$"), ("bn", "BatchNorm")]]
h2 = [plt.Line2D([], [], marker=m, ls="", color="k", mfc="none", ms=3.5, label=lab)
      for m, lab in [("o", "probe"), ("s", "probe+recon."), ("^", "autoenc.")]]
leg1 = ax.legend(handles=h, frameon=False, loc="upper right", handletextpad=0.1, borderaxespad=0.1, bbox_to_anchor=(1.0, 0.99), title="norm.", title_fontsize=5.8, fontsize=5.8, labelspacing=0.2)
ax.add_artist(leg1)
ax.legend(handles=h2, frameon=False, loc="upper right", handletextpad=0.1, borderaxespad=0.1, bbox_to_anchor=(1.0, 0.66), title="objective", title_fontsize=5.8, fontsize=5.8, labelspacing=0.2)
ax.annotate(r"$\ell_2$: $6.5\times$ vs. $0.56\times$", xy=(52, 6.4), xytext=(27, 7.9), fontsize=6.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"]))
ax.set_title("PMag window", fontsize=7.5)
print("PMag rho", rho, len(rows))

# (b) l2 temperature: Fisher ratio relative to no normalisation
rr = read_csv(f"{A2}/l2t/l2t_ratio_vs_none.csv")
ax = axes[1]
vars_ = [("s1", r"$\ell_2$, $s{=}1$"), ("s10", r"$\ell_2$, $s{=}10$"), ("learn10", r"$\ell_2$, learned")]
w = 0.36
for i, (v, lab) in enumerate(vars_):
    for j, (ds, col) in enumerate([("mnist", C["blue"]), ("fashion_mnist", C["orange"])]):
        r = [q for q in rr if q["dataset"] == ds and q["variant"] == v][0]
        ax.bar(i + (j - 0.5) * w, fnum(r["ratio_mean"]), w, yerr=fnum(r["ratio_std"]), color=col, capsize=1.5,
               error_kw=dict(lw=0.7), label=("MNIST" if ds == "mnist" else "Fashion-MNIST") if i == 0 else None)
ax.axhline(1, color="k", lw=0.6, ls="--")
ax.set_xticks(range(3))
ax.set_xticklabels([v[1] for v in vars_], fontsize=6.3)
ax.set_ylabel("Fisher ratio / no normalization")
ax.legend(frameon=False, loc="upper right", borderaxespad=0.1)
ax.set_title(r"$\ell_2$ temperature", fontsize=7.5)

# (c) redundancy with Fisher at several steps (69 configs; analysis_v2/ext2/mnist/step4_redundancy.csv)
rd = [r for r in read_csv(f"{A2}/ext2/mnist/step4_redundancy.csv") if r["set"].startswith("set_ii")]
ax = axes[2]
for mk, lab, col, ls_, m_ in [("h1_total", r"$H_1$ total", C["blue"], "-", "o"), ("pmag_fixed", "PMag", C["green"], "--", "s"),
                              ("mag_t1", r"Mag$(t{=}1)$", C["orange"], "-.", "^"), ("h1_top5", r"$H_1$ top-5", C["verm"], ":", "D")]:
    rr = sorted([r for r in rd if r["metric"] == mk], key=lambda r: int(r["step"]))
    xs = np.array([int(r["step"]) for r in rr]); ys = np.array([fnum(r["rho"]) for r in rr])
    lo = np.array([fnum(r["lo"]) for r in rr]); hi = np.array([fnum(r["hi"]) for r in rr])
    xo = xs * {"h1_total": 0.93, "pmag_fixed": 0.98, "mag_t1": 1.03, "h1_top5": 1.08}[mk]
    ax.errorbar(xo, ys, yerr=[ys - lo, hi - ys], color=col, ls=ls_, marker=m_, ms=3, lw=0.9, capsize=1.2, label=lab)
ax.axhline(0, color="k", lw=0.5)
ax.set_xscale("log")
ax.set_xticks([300, 1000, 5000, 15000]); ax.set_xticklabels(["300", "1k", "5k", "15k"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_ylim(-1.05, 1.3)
ax.set_xlabel("training step")
ax.set_ylabel(r"Spearman $\rho$ with Fisher")
ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=5.2, borderaxespad=0.1, handletextpad=0.3, columnspacing=0.8)
ax.set_title("redundancy with Fisher", fontsize=7.5)
for a, l in zip(axes, "abc"):
    panel_label(a, l, x=-0.28)
savefig(fig, "fig_caution.pdf")
