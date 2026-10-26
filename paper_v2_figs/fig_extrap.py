"""Appendix figure: increment of geometry (B1 -> B1+G, ridge) under interpolation (leave-one-config-out) and extrapolation splits.
Source: analysis_v2/ext2/{mnist,fashion}/step4_main_results.csv and step4_splits_results.csv (pooled folds)."""
from common import *

setup()
CUTS = [10, 30, 100, 300, 1000]
fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.95), gridspec_kw=dict(wspace=0.5))
SPL = [("loco", "leave-one-config-out", C["black"], "o", None), ("leave_one_n_out", "leave-one-$n$-out", C["blue"], "s", None),
       ("leave_one_wd_out", "leave-one-wd-out", C["green"], "^", None), ("leave_one_optimizer_out", r"Adam$\to$SGD", C["verm"], "D", None)]
for j, (model, tgt) in enumerate([("ridge", "acc_15kw"), ("ridge", "gap_15kw"), ("gbm", "acc_15kw"), ("gbm", "gap_15kw")]):
    ax = axes[j]
    for k, (sp, lab, col, mk, _) in enumerate(SPL):
        if sp == "loco":
            d = {int(r["cutoff"]): r for r in read_csv(f"{A2}/ext2/mnist/step4_main_results.csv")
                 if r["kind"] == "increment" and r["set"] == "B1+G" and r["base"] == "B1" and r["target"] == tgt and r["model"] == model}
        else:
            d = {int(r["cutoff"]): r for r in read_csv(f"{A2}/ext2/mnist/step4_splits_results.csv")
                 if r["split"] == sp and r["fold"] in ("POOLED",) and r["target"] == tgt and r["model"] == model}
        x = np.array(CUTS) * (0.88 + 0.08 * k)
        y = np.array([fnum(d[c]["delta"]) for c in CUTS])
        ax.plot(x, np.clip(y, -0.6, 0.6), color=col, marker=mk, ms=2.8, lw=0.8, label=lab)
        for xx, yy in zip(x, y):
            if abs(yy) > 0.6:
                ax.annotate("", xy=(xx, np.sign(yy) * 0.6), xytext=(xx, np.sign(yy) * 0.5), arrowprops=dict(arrowstyle="->", color=col, lw=0.6))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_ylim(-0.62, 0.62)
    ax.set_xlabel("cutoff step")
    ax.set_ylabel(r"$\Delta R^2$ ($B_1\!\to\! B_1{+}G$)")
    ax.set_title(f"{'ridge' if model == 'ridge' else 'GBM'}, {'test acc.' if tgt.startswith('acc') else 'gap'}", fontsize=7.5)
    if j == 0:
        ax.legend(frameon=False, loc="lower left", fontsize=5.5, handlelength=1.2, borderaxespad=0.1, labelspacing=0.2)
for a, l in zip(axes, "abcd"):
    panel_label(a, l, x=-0.3)
savefig(fig, "fig_extrap.pdf")
