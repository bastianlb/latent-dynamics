"""Figure: early-prediction R^2 vs cutoff, B1 vs B1+G (accuracy and gap); 305 runs, 69 configs."""
from common import *

setup()
rows = read_csv(f"{EXT}/step3_results.csv")
CUTS = [10, 30, 100, 300, 1000]


def get_ds(ds, target, model):
    return {int(r["cutoff"]): r for r in read_csv(f"{A2}/ext2/{ds}/step4_main_results.csv")
            if r["kind"] == "increment" and r["set"] == "B1+G" and r["base"] == "B1" and r["target"] == target and r["model"] == model}


def get(target, model, kind, s, base=""):
    out = {}
    for r in rows:
        if r["tag"] == "main" and r["target"] == target and r["model"] == model and r["kind"] == kind \
                and r["set"] == s and (r["base"] == base):
            out[int(r["cutoff"])] = r
    return out


fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.95), gridspec_kw=dict(wspace=0.5))
specs = [("acc_15kw", "", 0.03, None), ("gap_15kw", "", None, None)]
for j, (tgt, title, band_lo, band_hi) in enumerate(specs):
    axr, axd = axes[2 * j], axes[2 * j + 1]
    sets = [("B0", C["grey"], "--", r"$B_0$"), ("B0+G", C["grey"], "-", r"$B_0{+}G$"),
            ("B1", C["blue"], "-", r"$B_1$"), ("B1+G", C["verm"], "-", r"$B_1{+}G$"),
            ("G", C["green"], ":", r"$G$ alone")]
    for s, col, ls, lab in sets:
        d = get(tgt, "ridge", "set", s)
        axr.plot(CUTS, [fnum(d[c]["r2"]) for c in CUTS], ls=ls, color=col, marker="o", ms=2.5, label=lab)
    axr.set_xscale("log")
    axr.set_ylim(-0.05, 1.02) if tgt.startswith("acc") else axr.set_ylim(-0.2, 1.0)
    axr.set_xlabel("cutoff step")
    axr.set_ylabel(r"held-out $R^2$")
    axr.set_title(title if False else ("test accuracy" if j == 0 else "gap"), fontsize=7.5)
    if j == 0:
        axr.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.6, borderaxespad=0.1)
    # increments: MNIST ridge / GBM (filled), Fashion-MNIST ridge / GBM (open); equivalence band = equivalence margin
    marg = 0.03 if tgt.startswith("acc") else 0.05
    axd.axhspan(-marg, marg, color=C["green"], alpha=0.14, lw=0)
    for ds, model, col, off, mk, filled in [("mnist", "ridge", C["verm"], 0.84, "o", True), ("mnist", "gbm", C["purple"], 0.95, "s", True),
                                            ("fashion", "ridge", C["black"], 1.06, "o", False), ("fashion", "gbm", C["grey"], 1.19, "s", False)]:
        d = get_ds(ds, tgt, model)
        x = np.array(CUTS) * off
        y = np.array([fnum(d[c]["delta"]) for c in CUTS])
        lo = np.array([fnum(d[c]["lo95"]) for c in CUTS])
        hi = np.array([fnum(d[c]["hi95"]) for c in CUTS])
        axd.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk, color=col, mfc=(col if filled else "white"), ms=2.8, lw=0.8, capsize=1.2, mew=0.7,
                     label=f"{'MNIST' if ds == 'mnist' else 'Fashion'} {model.upper() if model == 'gbm' else 'ridge'}")
    axd.axhline(0, color="k", lw=0.6)
    axd.set_ylim((-0.12, 0.12) if tgt.startswith("acc") else (-0.2, 0.6))
    axd.set_xscale("log")
    axd.set_xlabel("cutoff step")
    axd.set_ylabel(r"$\Delta R^2$ ($B_1\!\to\! B_1{+}G$)")
    axd.set_title(("test accuracy" if j == 0 else "gap"), fontsize=7.5)
    if j == 1:
        axd.legend(frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.1, fontsize=5.8, labelspacing=0.25)
for ax, l in zip(axes, "abcd"):
    panel_label(ax, l, x=-0.3)
savefig(fig, "fig_predict.pdf")
