"""Appendix figure: Adam slingshots vs SGD (n=1000, wd=0), onset vs n."""
from common import *

setup()
runs = cache_runs()
fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.9), gridspec_kw=dict(wspace=0.55))
for tag, col, lab in [("sl40k_n1000_wd0", C["verm"], "Adam"), ("sgd40k_n1000_wd0", C["blue"], "SGD")]:
    d = np.load(runs[tag][0])
    s = d["step"].astype(float)
    ok = s >= 1
    axes[0].plot(s[ok], d["run/train_loss"][ok], color=col, label=lab, lw=0.9)
    axes[1].plot(s[ok], d["test/norm"][ok], color=col, lw=0.9)
    axes[2].plot(s[ok], d["test/fisher"][ok], color=col, lw=0.9)
for a in axes[:3]:
    a.set_xscale("log")
    a.set_xlabel("training step")
axes[0].set_yscale("log"); axes[0].set_ylabel("train loss")
axes[1].set_yscale("log"); axes[1].set_ylabel(r"mean latent norm $\|z\|$")
axes[2].set_ylabel("Fisher ratio (test probe)")
axes[0].legend(frameon=False, loc="lower left", borderaxespad=0.1)
rows = [r for r in read_csv(f"{DYN}/compare_adam_sgd/adam_vs_sgd.csv") if fnum(r["wd"]) == 0.0]
N = np.array([fnum(r["N"]) for r in rows])
on = np.array([fnum(r["adam_slingshot_onset_step_mean"]) for r in rows])
sd = np.array([fnum(r["adam_slingshot_onset_step_std"]) for r in rows])
ax = axes[3]
ax.errorbar(N, on, yerr=sd, fmt="o", color=C["verm"], capsize=2, ms=3.5, label="Adam: first event")
ax.axhline(40000, color=C["grey"], lw=0.6, ls=":")
ax.plot(N, [40000] * len(N), "x", color=C["blue"], ms=4, label="SGD: none by end (40k)")
ax.set_xscale("log")
from matplotlib.ticker import NullFormatter
ax.set_xticks([250, 1000, 4000]); ax.set_xticklabels(["250", "1000", "4000"]); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("training set size $n$")
ax.set_ylabel("event onset step")
ax.set_ylim(10000, 42000)
ax.legend(frameon=False, loc="lower right", borderaxespad=0.1, fontsize=6)
for a, l in zip(axes, "abcd"):
    panel_label(a, l, x=-0.3)
savefig(fig, "fig_sling.pdf")
