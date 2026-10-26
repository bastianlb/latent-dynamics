"""Page-1 teaser (draft 6; draft-5 version kept as fig_teaser_draft5.py; adds the pre-registered Fashion-MNIST study): train vs test probe latents of a small-n run, and exploratory vs pre-registered confirmatory increment of geometry over curves + design."""
from common import *
from matplotlib.gridspec import GridSpec

setup()
TAG = "phase_n250_wd1e-4"
runs = cache_runs()[TAG]
SEL = [5, 40, 20000]
CLASSES = [0, 1, 4, 7]
CCOL = [C["blue"], C["orange"], C["green"], C["purple"]]


fig = plt.figure(figsize=(PAGE_W, 3.3))
gs = GridSpec(2, 6, figure=fig, height_ratios=[0.9, 1.0], hspace=0.6, wspace=1.2)

# ---- top row: PCA of train (filled) and test (open) probe latents -------------------------------
run = f"mnist_probe_none_d16_s0_{TAG}"
cd = np.load(runs[0])
steps_all = cd["step"]
for k, st in enumerate(SEL):
    ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
    i = int(np.argmin(np.abs(steps_all - st)))
    s_act = int(steps_all[i])
    fn = f"step_{s_act:07d}.npz"
    te = np.load(f"{RUNS}/{run}/latents/{fn}")
    tr = np.load(f"{RUNS}/{run}/latents_train/{fn}")
    Z = np.concatenate([te["latents"], tr["latents"]], 0)
    mu = Z.mean(0)
    # PCA on test+train jointly
    U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    P = Vt[:2].T
    zt, zr = (te["latents"] - mu) @ P, (tr["latents"] - mu) @ P
    sc = np.sqrt((np.concatenate([zt, zr]) ** 2).sum(1)).max()
    zt, zr = zt / sc, zr / sc
    for c, col in zip(CLASSES, CCOL):
        mt, mr = te["labels"] == c, tr["labels"] == c
        ax.scatter(zr[mr, 0], zr[mr, 1], s=7, color=col, lw=0, alpha=0.9)
        ax.scatter(zt[mt, 0], zt[mt, 1], s=9, facecolor="none", edgecolor=col, lw=0.5, marker="D", alpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    tra, tea = float(cd["run/train_acc"][i]), float(cd["run/test_acc"][i])
    ax.set_title(f"step {s_act}: train {tra:.2f}, test {tea:.2f}", fontsize=7)
    ax.set_xlabel("PCA of probe latents", fontsize=6.5, labelpad=2)
    if k == 1:
        h = [plt.Line2D([], [], marker="o", ls="", color="k", ms=2.5, label="train probe"),
             plt.Line2D([], [], marker="D", ls="", mfc="none", mec="k", ms=2.8, mew=0.5, label="test probe")]
        ax.legend(handles=h, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6, handletextpad=0.1, columnspacing=1.0)
panel_label(fig.axes[0], "a", x=-0.12, y=1.12)

# ---- bottom row: exploratory vs confirmatory Delta R^2 (ridge, B1 -> B1 + geometry, 90% config-bootstrap CI) -----------
E3M = f"{A2}/ext3/mnist/margin_mstar.csv"
CFR = f"{A2}/confirm/results/primary_ridge.csv"
CFF = f"{A2}/confirm_fm/results/primary_ridge.csv"
ex = {(r["target"], int(r["cutoff"])): r for r in read_csv(E3M) if r["model"] == "ridge"}
cf = {(r["X"], r["target"], int(r["cutoff"])): r for r in read_csv(CFR)}
fm = {(r["X"], r["target"], int(r["cutoff"])): r for r in read_csv(CFF)}
cuts = [10, 30, 100, 300, 1000]
for k, (tgt, mg, ttl, lab) in enumerate([("acc_15kw", 0.03, "test accuracy", "b"), ("gap_15kw", 0.05, "generalization gap", "c")]):
    ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
    ax.axhspan(-mg, mg, color=C["green"], alpha=0.14, lw=0, label="equivalence margin")
    ax.axhline(0, color="k", lw=0.6)
    xs = np.arange(len(cuts))
    for j, c in enumerate(cuts):
        r = ex[(tgt, c)]
        y, lo, hi = fnum(r["delta"]), fnum(r["lo90_iid"]), fnum(r["hi90_iid"])
        ax.errorbar([j - 0.24], [y], yerr=[[y - lo], [hi - y]], fmt="o", color=C["blue"], ms=3.2, lw=1.0, capsize=1.8,
                    label=("exploratory MNIST grid (305 runs)" if j == 0 else None))
        r = cf[("G", tgt, c)]
        y, lo, hi = fnum(r["delta"]), fnum(r["lo90"]), fnum(r["hi90"])
        ax.errorbar([j], [y], yerr=[[y - lo], [hi - y]], fmt="s", color=C["verm"], ms=3.2, lw=1.0, capsize=1.8,
                    label=("pre-registered MNIST (84 runs)" if j == 0 else None))
        r = fm[("G", tgt, c)]
        y, lo, hi = fnum(r["delta"]), fnum(r["lo90"]), fnum(r["hi90"])
        ax.errorbar([j + 0.24], [y], yerr=[[y - lo], [hi - y]], fmt="D", color=C["orange"], ms=3.0, lw=1.0, capsize=1.8,
                    label=("pre-registered Fashion-MNIST (90 runs)" if j == 0 else None))
    ax.set_xticks(xs); ax.set_xticklabels([str(c) for c in cuts])
    ax.set_xlim(-0.6, len(cuts) - 0.4)
    ax.set_ylim(-0.29, 0.17)
    ax.set_xlabel("cutoff (training step)")
    if k == 0:
        ax.set_ylabel(r"$\Delta R^2$ over curves + design")
        hh, ll = ax.get_legend_handles_labels()
    ax.set_title(ttl, fontsize=7)
    panel_label(ax, lab, x=-0.16 if k == 0 else -0.12, y=1.08)
fig.legend(hh, ll, loc="lower center", bbox_to_anchor=(0.5, -0.085), ncol=4, frameon=False, fontsize=6.2, handletextpad=0.2, columnspacing=1.2)
savefig(fig, "fig_teaser.pdf")
