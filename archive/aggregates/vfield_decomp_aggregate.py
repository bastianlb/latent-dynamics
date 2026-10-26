"""Aggregate vfield_decomp per-run npz: summary csv + figures. Usage: python vfield_decomp_aggregate.py [--out DIR]"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import argparse, glob, csv, re, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/decomp"
BANDS = [(1, 10), (10, 100), (100, 1000), (1000, 10 ** 9)]; BN = ["b1_10", "b10_100", "b100_1000", "b1000_up"]
PN = ["R1_affine", "R2_classaffine", "R3_knn", "R3_krr", "J_z", "J_zy", "R3_knn_null", "R3_krr_null", "J_zy_null_class"]


def parse(fn):
    b = os.path.basename(fn)[:-4]; probe = "train" if b.endswith("_train") else "test"; rid = b[: -len(probe) - 1]
    m = re.search(r"_s(\d+)_(phase|gap)_n(\w+?)_(wd|ln)([\d.e-]+)$", rid); seed, fam, N, kind, val = m.groups()
    return rid, probe, int(seed), fam, (60000 if N == "full" else int(N)), (float(val) if kind == "wd" else 0.0), (float(val) if kind == "ln" else 0.0)


def bm(x, st, lo, hi):
    k = (st >= lo) & (st < hi); return float(np.nanmedian(x[k])) if k.any() else np.nan


def pcross(diff, steps, frac=0.8):
    for k in range(len(diff)):
        if diff[k] > 0 and np.nanmean(diff[k:] > 0) >= frac: return float(steps[k])
    return np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args(); out = a.out; os.makedirs(f"{out}/figures", exist_ok=True)
    R = []
    for f in sorted(glob.glob(f"{out}/runs/*.npz")):
        rid, probe, seed, fam, N, wd, ln = parse(f); z = np.load(f, allow_pickle=True); R.append(dict(rid=rid, probe=probe, seed=seed, fam=fam, N=N, wd=wd, ln=ln, z=z, st=z["steps"]))
    print(len(R), "run-probes"); rows = []
    for r in R:
        z, st = r["z"], r["st"]; s1 = st[1:]; P = z["part_mean"]; row = dict(run_id=r["rid"], probe=r["probe"], seed=r["seed"], family=r["fam"], N=r["N"], wd=r["wd"], ln=r["ln"], n=int(z["n"]), min_class=int(z["min_class"]))
        for (lo, hi), bn in zip(BANDS, BN):
            v = {n: bm(P[i], s1, lo, hi) for i, n in enumerate(PN)}
            for n, x in v.items(): row[f"{n}_{bn}"] = x
            row[f"unexplained_krr_{bn}"] = 1 - v["R3_krr"]; resid = 1 - v["R1_affine"]
            row[f"resid_frac_classaffine_{bn}"] = (v["R2_classaffine"] - v["R1_affine"]) / resid if resid > 0 else np.nan
            row[f"resid_frac_nonlin_krr_{bn}"] = (v["R3_krr"] - v["R2_classaffine"]) / resid if resid > 0 else np.nan
            row[f"resid_frac_nonlin_knn_{bn}"] = (v["R3_knn"] - v["R2_classaffine"]) / resid if resid > 0 else np.nan
            row[f"resid_frac_joint_zy_{bn}"] = (v["J_zy"] - v["R1_affine"]) / resid if resid > 0 else np.nan
            row[f"resid_frac_joint_z_{bn}"] = (v["J_z"] - v["R1_affine"]) / resid if resid > 0 else np.nan
            row[f"resid_frac_null_krr_{bn}"] = (v["R3_krr_null"] - v["R2_classaffine"]) / resid if resid > 0 else np.nan
            for ci, c in enumerate(z["gen_cols"]): row[f"gen_{c}_{bn}"] = bm(z["gen"][:, ci], s1, lo, hi)
            row[f"gen_anisotropy_{bn}"] = row[f"gen_shear_norm_{bn}"] / (abs(row[f"gen_iso_expansion_{bn}"]) + 1e-30)
            for k in z.files:
                if k.startswith("nl_") and z[k].ndim == 1 and len(z[k]) == len(s1): row[f"{k}_{bn}"] = bm(z[k], s1, lo, hi)
        row["gen_fallback_frac"] = float(np.mean(z["gen_fallback"]))
        if "nl_mis_f1_cos_given" in z.files:
            for fn in ("f1", "f2"):
                d = z[f"nl_mis_{fn}_cos_given"] - z[f"nl_mis_{fn}_cos_true"]; row[f"mis_{fn}_cross_step"] = pcross(d, s1); row[f"mis_{fn}_cross_epoch"] = pcross(d, s1) * 128.0 / r["N"]
                row[f"mis_{fn}_diff_early"] = bm(d, s1, 1, 100); row[f"mis_{fn}_diff_late"] = bm(d, s1, 1000, 10 ** 9)
        rows.append(row)
    cols = []
    for d in rows:
        for c in d:
            if c not in cols: cols.append(c)
    with open(f"{out}/summary_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    ph = [r for r in R if r["fam"] == "phase" and r["probe"] == "test"]
    cells = [(250, 0.0), (1000, 0.0), (4000, 0.0), (4000, 1e-2), (60000, 0.0), (60000, 3e-2)]
    # --- partition stacked / residual fractions vs step
    if ph:
        fig, ax = plt.subplots(2, 3, figsize=(17, 8)); ax = ax.ravel()
        for a_, (N, wd) in zip(ax, cells):
            sel = [r for r in ph if r["N"] == N and r["wd"] == wd]
            if not sel: continue
            s1 = sel[0]["st"][1:]; M = np.nanmean(np.stack([r["z"]["part_mean"] for r in sel]), 0); R1, R2, R3 = M[0], M[1], M[3]
            a_.stackplot(s1, np.clip(R1, 0, 1), np.clip(R2 - R1, 0, 1), np.clip(R3 - R2, 0, 1), np.clip(1 - R3, 0, 1), labels=["global affine", "+ class-wise affine", "+ nonlinear (krr)", "unexplained"], colors=["C0", "C1", "C2", "0.8"])
            a_.set_xscale("log"); a_.set_ylim(0, 1); a_.set_title(f"N={N} wd={wd:g}: held-out partition of sum||d||^2 (mean over {len(sel)} seeds)", fontsize=8)
        ax[0].legend(fontsize=6, loc="lower left"); fig.tight_layout(); fig.savefig(f"{out}/figures/partition_stacked.png", dpi=60); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.8)); col = dict(R2_classaffine="C1", R3_knn="C2", R3_krr="C3", J_z="C4", J_zy="C5", R3_krr_null="gray", R3_knn_null="k", J_zy_null_class="C6")
        for a_, (N, wd) in zip(ax, [(250, 0.0), (4000, 0.0), (4000, 1e-2)]):
            sel = [r for r in ph if r["N"] == N and r["wd"] == wd]
            if not sel: continue
            s1 = sel[0]["st"][1:]; M = np.nanmean(np.stack([r["z"]["part_mean"] for r in sel]), 0); R1 = M[0]
            for i, n in enumerate(PN):
                if n == "R1_affine": continue
                a_.plot(s1, (M[i] - (M[1] if n.startswith("R3") and "null" in n else R1)) / np.maximum(1 - R1, 1e-9), c=col[n], label=n, ls=":" if "null" in n else "-")
            a_.set_xscale("log"); a_.set_ylim(-0.5, 1); a_.axhline(0, c="k", lw=.4); a_.set_title(f"N={N} wd={wd:g}: fraction of the post-affine residual variance explained", fontsize=8)
        ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{out}/figures/residual_fractions.png", dpi=60); plt.close(fig)
        # --- generator time series
        fig, ax = plt.subplots(2, 3, figsize=(17, 8)); gi = {c: i for i, c in enumerate(ph[0]["z"]["gen_cols"])}
        for k, (name, sc) in enumerate((("iso_expansion", "symlog"), ("shear_norm", "log"), ("rotation_norm", "log"))):
            for (N, wd) in cells:
                sel = [r for r in ph if r["N"] == N and r["wd"] == wd]
                if not sel: continue
                s1 = sel[0]["st"][1:]; arr = np.stack([r["z"]["gen"][:, gi[name]] for r in sel]); ax[0, k].plot(s1, np.nanmedian(arr, 0), label=f"N{N} wd{wd:g}")
            ax[0, k].set_xscale("log"); ax[0, k].set_yscale(sc, **({"linthresh": 1e-7} if sc == "symlog" else {})); ax[0, k].set_title(f"{name} of L_t = logm(A_t)/dstep (median over seeds)", fontsize=8)
        for k, name in enumerate(("sym_eig_max", "sym_eig_min", "log_cond_A")):
            for (N, wd) in cells:
                sel = [r for r in ph if r["N"] == N and r["wd"] == wd]
                if not sel: continue
                s1 = sel[0]["st"][1:]; arr = np.stack([r["z"]["gen"][:, gi[name]] for r in sel]); ax[1, k].plot(s1, np.nanmedian(arr, 0))
            ax[1, k].set_xscale("log"); ax[1, k].set_yscale("symlog", linthresh=1e-7) if k < 2 else None; ax[1, k].set_title(name, fontsize=8)
        ax[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{out}/figures/generator_series.png", dpi=60); plt.close(fig)
        # --- nonlinear field series
        keys = [("nl_f2_spearman_margin", "Spearman(|f2|, NCM margin)"), ("nl_f2_energy_share_lowest20pct_margin", "energy share of lowest-margin 20%"), ("nl_f1_cos_own", "cos(f1, to own mean)"), ("nl_f1_cos_nearest_other", "cos(f1, to nearest other mean)"),
                ("nl_f2_div_median", "median local divergence of f2"), ("nl_f1_n90", "n90 components of f1")]
        fig, ax = plt.subplots(2, 3, figsize=(17, 8)); ax = ax.ravel()
        for a_, (k, lab) in zip(ax, keys):
            for (N, wd) in cells:
                sel = [r for r in ph if r["N"] == N and r["wd"] == wd and k in r["z"].files]
                if not sel: continue
                a_.plot(sel[0]["st"][1:], np.nanmedian(np.stack([r["z"][k] for r in sel]), 0), label=f"N{N} wd{wd:g}")
            a_.set_xscale("log"); a_.set_title(lab, fontsize=8)
        ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{out}/figures/nonlinear_field_series.png", dpi=60); plt.close(fig)
        # --- heatmaps over (N, wd)
        Ns = sorted({r["N"] for r in ph}); wds = sorted({r["wd"] for r in ph})
        pairs = [("resid_frac_classaffine_b100_1000", "class-affine share of post-affine resid (steps 100-1000)"), ("resid_frac_nonlin_krr_b100_1000", "nonlinear (krr) share"), ("resid_frac_joint_zy_b100_1000", "joint krr_zy share"), ("R1_affine_b100_1000", "global affine R2"),
                 ("gen_gen_anisotropy_b100_1000", "L anisotropy |shear|/|iso|"), ("gen_iso_expansion_b100_1000", "iso expansion rate")]
        fig, ax = plt.subplots(2, 3, figsize=(16, 9)); ax = ax.ravel()
        for a_, (k, lab) in zip(ax, pairs):
            k2 = k.replace("gen_gen_", "gen_"); M = np.full((len(Ns), len(wds)), np.nan)
            for i, N in enumerate(Ns):
                for j, wd in enumerate(wds):
                    v = [x[k2] for x, rr in zip(rows, R) if rr["fam"] == "phase" and rr["probe"] == "test" and rr["N"] == N and rr["wd"] == wd and k2 in x]
                    if v: M[i, j] = np.nanmean(v)
            im = a_.imshow(M, origin="lower", aspect="auto", cmap="viridis"); a_.set_title(lab, fontsize=8); a_.set_xticks(range(len(wds))); a_.set_xticklabels([f"{w:g}" for w in wds], fontsize=6); a_.set_yticks(range(len(Ns))); a_.set_yticklabels(Ns, fontsize=6); plt.colorbar(im, ax=a_, fraction=.046)
            for i in range(len(Ns)):
                for j in range(len(wds)):
                    if np.isfinite(M[i, j]): a_.text(j, i, f"{M[i,j]:.2g}", ha="center", va="center", fontsize=6, color="w")
        fig.tight_layout(); fig.savefig(f"{out}/figures/heatmaps.png", dpi=60); plt.close(fig)
    # --- mislabeled f-only
    gp = [r for r in R if r["fam"] == "gap" and r["probe"] == "train" and "nl_mis_f1_cos_given" in r["z"].files]
    if gp:
        cm = {250: "C0", 1000: "C1", 4000: "C2", 16000: "C3", 60000: "C4"}; fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        for j, fn in enumerate(("f1", "f2")):
            for r in gp:
                ep = r["st"][1:] * 128.0 / r["N"]; d = r["z"][f"nl_mis_{fn}_cos_given"] - r["z"][f"nl_mis_{fn}_cos_true"]; ax[0 if r["ln"] < .3 else 1, j].plot(ep, d, c=cm[r["N"]], lw=.7, label=f"N{r['N']}" if r["seed"] == 0 else None)
            for k in (0, 1): ax[k, j].axhline(0, c="k", lw=.5); ax[k, j].set_xscale("log"); ax[k, j].set_title(f"{fn} only (global affine removed{', + class affine' if fn=='f2' else ''}): cos(f,->given) - cos(f,->true), ln={0.2 if k==0 else 0.4}", fontsize=8)
        ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{out}/figures/mislabeled_f_only.png", dpi=60); plt.close(fig)
    print("done")


if __name__ == "__main__": main()
