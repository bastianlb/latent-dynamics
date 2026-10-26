"""Page-1 teaser: train vs test probe latents of a small-n run, divergence timing, and early-prediction increment."""
from common import *
from matplotlib.gridspec import GridSpec

setup()
TAG = "phase_n250_wd1e-4"
runs = cache_runs()[TAG]
SEL = [5, 40, 20000]
CLASSES = [0, 1, 4, 7]
CCOL = [C["blue"], C["orange"], C["green"], C["purple"]]


fig = plt.figure(figsize=(PAGE_W, 3.2))
gs = GridSpec(2, 6, figure=fig, height_ratios=[0.9, 1.0], hspace=0.55, wspace=1.2)

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

# ---- bottom-left: normalised progress of gap and train-test divergences ---------------------------
ax = fig.add_subplot(gs[1, 0:3])
steps = cd["step"].astype(float)
seeds = sorted(runs)
D = {}


def series(d, kind):
    if kind == "gap":
        return d["run/train_acc"] - d["run/test_acc"]
    return d[f"train/{kind}"] - d[f"test/{kind}"]


for kind in ["gap", "eff_rank_pr", "mag_t1", "fisher"]:
    D[kind] = np.nanmean([series(np.load(runs[s]), kind) for s in seeds], axis=0)
masks = np.load(runs[0])["topo_mask"]
spec = [("gap", "accuracy gap", C["black"], "-", 1.6),
        ("eff_rank_pr", "eff. rank", C["blue"], "-", 1.1),
        ("mag_t1", r"Mag$(t{=}1)$", C["green"], "--", 1.1),
        ("fisher", "Fisher", C["verm"], "-.", 1.1)]
for kind, lab, col, ls, lw in spec:
    v = D[kind].copy()
    ok = np.isfinite(v) & (steps >= 1)
    if kind in ("mag_t1",):
        ok &= masks
    x0 = v[0]
    p = (v - x0) / (v[-1] - x0)
    ax.plot(steps[ok], p[ok], color=col, ls=ls, lw=lw, label=lab)
ax.axhline(0.5, color=C["grey"], lw=0.5, ls=":")
ax.set_xscale("log")
ax.set_ylim(-0.3, 2.7)
ax.set_xlim(1, 2e4)
ax.set_xlabel("training step")
ax.set_ylabel(r"progress of train$-$test $\Delta$")
ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.8, columnspacing=0.8, borderaxespad=0.1)
ax.set_title(r"descriptive timing, $n{=}250$ run (3 seeds)", fontsize=7)
panel_label(ax, "b", x=-0.14, y=1.1)

# ---- bottom-right: Delta R^2 of geometry beyond curves + design ----------------------------------
ax = fig.add_subplot(gs[1, 3:6])
rows = read_csv(f"{EXT}/step3_results.csv")
cuts = [10, 100, 1000]
mks = {10: "o", 100: "s", 1000: "^"}
for xi, tgt in enumerate(["acc_15kw", "gap_15kw"]):
    for j, c in enumerate(cuts):
        r = [q for q in rows if q["tag"] == "main" and q["target"] == tgt and q["model"] == "ridge" and
             q["kind"] == "increment" and q["set"] == "B1+G" and q["base"] == "B1" and int(q["cutoff"]) == c][0]
        y, lo, hi = fnum(r["delta"]), fnum(r["ci_lo"]), fnum(r["ci_hi"])
        xx = xi + (j - 1) * 0.22
        ax.errorbar([xx], [y], yerr=[[y - lo], [hi - y]], fmt=mks[c], color=C["verm"], ms=3.8, lw=1.0, capsize=2,
                    label=(f"cutoff step {c}" if xi == 0 else None))
for xi, mg in enumerate([0.03, 0.05]):
    ax.fill_between([xi - 0.42, xi + 0.42], -mg, mg, color=C["green"], alpha=0.16, lw=0, label=("equivalence margin" if xi == 0 else None))
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["test accuracy", "generalization gap"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-0.1, 0.25)
ax.set_ylabel(r"$\Delta R^2$ over curves + design")
ax.legend(frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.1)
ax.set_title("305 runs, ridge, 95% CI; targets: 12k\u201318k median", fontsize=6.5)
panel_label(ax, "c", x=-0.22, y=1.1)
savefig(fig, "fig_teaser.pdf")
