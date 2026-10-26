"""Fig 8 of the dynamics report: AE vs classifier latent-field comparison (CPU only).
(a) late R1 vs N (per-run band metric R1_b1000_up, mean +- sd over 3 seeds; same loaders as ae_lds_report.py)
(b) late group-wise share (R2-R1)/(1-R1), labels vs k-means(10) groups, from summary.csv columns vf_te_{lab,km}_classaffine_gain_late
(c) final kNN agreement of test latents at full N for label groups vs input-cluster groups."""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]: os.environ.setdefault(v, "8")
import csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ae_lds_load as L
A = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis"; OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_report/figures"
CL, AE, LIN, PIX = "#d62728", "#1f77b4", "#7f7f7f", "#c8c8c8"
NS = [250, 1000, 4000, 60000]; NL = ["250", "1k", "4k", "full"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
ms = lambda v: (float(np.mean(v)), float(np.std(v))) if len(v) else (np.nan, np.nan)

def rows_csv(p):
    return list(csv.DictReader(open(p)))
def fl(x):
    try: return float(x)
    except Exception: return np.nan

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4), gridspec_kw={"width_ratios": [1, 1.15, 1.25]})
# ---- (a)
a = ax[0]
clf = {ds: L.load_classifier(ds, "test") for ds in ("mnist", "fashion_mnist")}
ae = L.load_ae("lab")
for ds, ls, lab in (("mnist", "-", "MNIST"), ("fashion_mnist", "--", "FM")):
    for nm, col, rows in (("classifier", CL, [r for r in clf[ds] if r["wd"] == 0.0]),
                          ("MLP AE", AE, [r for r in ae if r["dataset"] == ds and r["probe"] == "test" and r["arch"] == "mlp" and r["d"] == 16 and r["wd"] == 0.0]),
                          ("linear AE", LIN, [r for r in ae if r["dataset"] == ds and r["probe"] == "test" and r["arch"] == "linear" and r["d"] == 16 and r["wd"] == 0.0])):
        m, s = [], []
        for N in NS:
            v = [r["R1_b1000_up"] for r in rows if r["N"] == N]; mm, ss = ms(v); m.append(mm); s.append(ss)
        a.errorbar(range(4), m, yerr=s, color=col, ls=ls, marker="o", ms=3.5, lw=1.4, capsize=2, label=f"{nm}, {lab}")
a.set_xticks(range(4)); a.set_xticklabels(NL); a.set_xlabel("training-set size $N$"); a.set_ylabel("late $R_1$ (steps $\\geq$ 1000)"); a.set_ylim(0.3, 1.02); a.grid(alpha=0.3)
a.set_title("(a) Late global-affine share", fontsize=9); a.legend(fontsize=6, ncol=2, loc="lower left", frameon=False)
# ---- (b)
b = ax[1]
SH = {"mnist": rows_csv(f"{A}/summary.csv"), "fashion_mnist": rows_csv(f"{A}/fashion/summary.csv")}
pos = 0; ticks = []; tl = []
for di, ds in enumerate(("mnist", "fashion_mnist")):
    for ni, N in enumerate(NS):
        for fam, col, dx in (("probe", CL, -0.17), ("AE", AE, 0.17)):
            rr = [r for r in SH[ds] if r["family"] == fam and r["latent_dim"] == "16" and fl(r["weight_decay"]) == 0 and int(r["N"]) == N]
            for key, filled, off in (("lab", True, -0.06), ("km", False, 0.06)):
                v = [fl(r[f"vf_te_{key}_classaffine_gain_late"]) for r in rr]; mm, ss = ms(v)
                b.errorbar(pos + dx + off, mm, yerr=ss, color=col, marker="o", ms=5, mfc=col if filled else "white", capsize=2, lw=1)
        ticks.append(pos); tl.append(NL[ni]); pos += 1
    pos += 0.8
b.set_xticks(ticks); b.set_xticklabels(tl); b.set_xlabel("training-set size $N$"); b.set_ylabel("group-wise share of residual")
b.text(1.5, b.get_ylim()[1] * 0.98, "MNIST", ha="center", va="top", fontsize=8); b.text(6.3, b.get_ylim()[1] * 0.98, "FM", ha="center", va="top", fontsize=8)
b.grid(alpha=0.3); b.set_title("(b) Class-wise increment", fontsize=9)
from matplotlib.lines import Line2D
b.legend(handles=[Line2D([], [], color=CL, marker="o", ls="", label="classifier"), Line2D([], [], color=AE, marker="o", ls="", label="MLP AE"),
                  Line2D([], [], color="k", marker="o", ls="", label="label groups"), Line2D([], [], color="k", marker="o", mfc="white", ls="", label="$k$-means groups")], fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.9), frameon=False)
# ---- (c)
c = ax[2]
SC = {"MNIST": rows_csv(f"{A}/summary.csv"), "FM": rows_csv(f"{A}/fashion/summary.csv"), "CIFAR-10": rows_csv(f"{A}/cifar10/summary.csv")}
Nfull = {"MNIST": 60000, "FM": 60000, "CIFAR-10": 50000}
x = 0; dsx = []; w = 0.17
for ds, rr in SC.items():
    full = [r for r in rr if int(r["N"]) == Nfull[ds] and fl(r["weight_decay"]) == 0]
    ddim = "64" if ds == "CIFAR-10" else "16"
    aer = [r for r in full if r["family"] == "AE" and r["latent_dim"] == ddim]
    items = [("raw pixels", PIX, [fl(r["te_lab_knn_pixel"]) for r in aer], [fl(r["te_km_knn_pixel"]) for r in aer]),
             ("linear AE", LIN, *[[fl(r[k]) for r in full if r["family"] == "linAE"] for k in ("final_te_lab_knn", "final_te_km_knn")]),
             ("MLP AE" if ds != "CIFAR-10" else "conv AE", AE, *[[fl(r[k]) for r in aer] for k in ("final_te_lab_knn", "final_te_km_knn")]),
             ("classifier", CL, *[[fl(r[k]) for r in full if r["family"] == "probe" and r["latent_dim"] == "16"] for k in ("final_te_lab_knn", "final_te_km_knn")])]
    x0 = x
    for nm, col, lab, km in items:
        for j, (v, hatch) in enumerate(((lab, None), (km, "////"))):
            if len(v):
                mm, ss = ms(v); c.bar(x + (j - 0.5) * w, mm, w, yerr=ss, color=col, edgecolor="k" if hatch else col, hatch=hatch, lw=0.4, capsize=1.5, label=nm if (ds == "MNIST" and j == 0) else None)
            elif nm == "classifier":
                c.text(x, 0.02, "n/a", rotation=90, ha="center", va="bottom", fontsize=6, color=CL)
        x += 2 * w + 0.06
    dsx.append(((x0 + x - 0.06 - 2 * w) / 2 + w * 0.0, ds)); x += 0.5
c.set_xticks([d[0] + w * 0.0 for d in dsx]); c.set_xticklabels([d[1] for d in dsx]); c.set_ylabel("kNN agreement (test latents)"); c.set_ylim(0, 1.05); c.grid(alpha=0.3, axis="y")
c.set_title("(c) Final class structure, full $N$", fontsize=9)
h, l = c.get_legend_handles_labels()
h += [plt.Rectangle((0, 0), 1, 1, fc="white", ec="k"), plt.Rectangle((0, 0), 1, 1, fc="white", ec="k", hatch="////")]; l += ["label groups", "input clusters"]
c.legend(h, l, fontsize=6, ncol=3, loc="upper center", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig8_field_compare.pdf"); fig.savefig(f"{OUT}/fig8_field_compare.png", dpi=150)
# printed data summary
print("done")
