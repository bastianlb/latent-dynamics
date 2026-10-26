"""Aggregate vfield per-run npz into summary csv/figures. Usage: python vfield_aggregate.py [--out DIR]"""
import os, sys
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, csv, re, pickle, collections, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield"
BANDS = [(1, 10), (10, 100), (100, 1000), (1000, 10 ** 9)]; BN = ["b1_10", "b10_100", "b100_1000", "b1000_up"]
KEYM = ["M1", "M2", "knn10", "knn10y", "mlp", "null_knn_shuffled", "null_time_M2", "null_time_knn10y"]
PC = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/perclass/cache"; RUNS = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs"


def parse(fn):
    b = os.path.basename(fn)[:-4]; probe = "train" if b.endswith("_train") else "test"; rid = b[: -len(probe) - 1]
    m = re.search(r"_s(\d+)_(phase|gap)_n(\w+?)_(wd|ln)([\d.e-]+)$", rid); seed, fam, N, kind, val = m.groups()
    N = 60000 if N == "full" else int(N); return rid, probe, int(seed), fam, N, (float(val) if kind == "wd" else 0.0), (float(val) if kind == "ln" else 0.0)


def band_med(x, st, lo, hi):
    k = (st >= lo) & (st < hi)
    return float(np.nanmedian(x[k])) if k.any() else np.nan


def persistent_cross(diff, steps, frac=0.8):
    """first step from which diff>0 for >= frac of the remaining intervals (sign flip toward the given label)."""
    for k in range(len(diff)):
        if diff[k] > 0 and np.nanmean(diff[k:] > 0) >= frac: return float(steps[k])
    return np.nan


def mem_cross_perclass(rid):
    p = f"{PC}/v2_{rid}.pkl"
    if not os.path.isfile(p): return np.nan
    r = pickle.load(open(p, "rb")); P = r.get("train")
    if P is None or "mem_fit_c" not in P: return np.nan
    st = P["steps"]; out = []
    for c in range(10):
        ok = np.nan_to_num(P["mem_fit_c"][:, c]) > np.nan_to_num(P["mem_cor_c"][:, c])
        if not ok[-1]: continue
        bad = np.where(~ok)[0]; out.append(float(st[0 if len(bad) == 0 else bad[-1] + 1]))
    return float(np.median(out)) if out else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args(); out = a.out; os.makedirs(f"{out}/figures", exist_ok=True)
    files = sorted(glob.glob(f"{out}/runs/*.npz")); R = []
    for f in files:
        rid, probe, seed, fam, N, wd, ln = parse(f); z = np.load(f, allow_pickle=True); R.append(dict(rid=rid, probe=probe, seed=seed, fam=fam, N=N, wd=wd, ln=ln, z=z, steps=z["steps"], models=list(z["models"]), frames=list(z["frames"])))
    print(len(R), "run-probes")
    rows = []
    for r in R:
        z, st = r["z"], r["steps"]; s1 = st[1:]; row = dict(run_id=r["rid"], probe=r["probe"], seed=r["seed"], family=r["fam"], N=r["N"], wd=r["wd"], ln=r["ln"], n_samples=int(z["n"]), min_class=int(z["min_class"]), n_classes=int(z["n_classes"]))
        R2 = z["R2c_mean"]
        for mi, m in enumerate(r["models"]):
            for fi, fr in enumerate(r["frames"]):
                for (lo, hi), bn in zip(BANDS, BN): row[f"R2_{m}_{fr}_{bn}"] = band_med(R2[mi, fi], s1, lo, hi)
                row[f"R2_{m}_{fr}_all10"] = band_med(R2[mi, fi], s1, 10, 10 ** 9)
        row["R20_M0_raw_all10"] = band_med(z["R20_mean"][0, 0], s1, 10, 10 ** 9)
        for fi, fr in enumerate(r["frames"]):
            row[f"persist_{fr}_all10"] = band_med(z["persistence"][fi], st[1:-1], 10, 10 ** 9); row[f"persist_null_{fr}_all10"] = band_med(z["persistence_null"][fi], st[1:-1], 10, 10 ** 9)
        row["ufm_cos_last10"] = float(np.nanmean(z["ufm_cos"][-10:])); row["ufm_cos_all10"] = band_med(z["ufm_cos"], s1, 10, 10 ** 9)
        dstep = np.diff(st).astype(float)
        for fr in ("raw", "affine"):
            row[f"fp_dist_{fr}_all10"] = band_med(np.nanmedian(z[f"char_{fr}_fp_dist_over_sep"], 1), s1, 10, 10 ** 9)
            row[f"maxre_per_step_{fr}_all10"] = band_med(np.nanmedian(z[f"char_{fr}_max_re"], 1) / dstep, s1, 10, 10 ** 9)
            row[f"trace_per_step_{fr}_all10"] = band_med(np.nanmedian(z[f"char_{fr}_trace"], 1) / dstep, s1, 10, 10 ** 9)
            for (lo, hi), bn in zip(BANDS, BN): row[f"fp_dist_{fr}_{bn}"] = band_med(np.nanmedian(z[f"char_{fr}_fp_dist_over_sep"], 1), s1, lo, hi)
        if r["probe"] == "train" and r["ln"] > 0 and "mis_affine_cos_given" in z.files:
            epochs = st[1:] * 128.0 / r["N"]; pm = mem_cross_perclass(r["rid"])
            for fr in ("raw", "procrustes", "affine"):
                d = z[f"mis_{fr}_cos_given"] - z[f"mis_{fr}_cos_true"]; row[f"vf_cross_step_{fr}"] = persistent_cross(d, s1); row[f"mis_diff_early_{fr}"] = band_med(d, s1, 1, 100); row[f"mis_diff_late_{fr}"] = band_med(d, s1, 1000, 10 ** 9)
            row["perclass_mem_cross_step"] = pm; row["perclass_mem_cross_epoch"] = pm * 128.0 / r["N"] if np.isfinite(pm) else np.nan; row["n_bad"] = int(z["n_bad"])
            row["vf_cross_epoch_affine"] = row["vf_cross_step_affine"] * 128.0 / r["N"]
        rows.append(row)
    cols = []
    for d in rows:
        for c in d:
            if c not in cols: cols.append(c)
    with open(f"{out}/summary_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # --- figure 1: R2 vs log step, phase grid, test probe, medians over runs
    ph = [r for r in R if r["fam"] == "phase" and r["probe"] == "test"]
    if ph:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5)); st = ph[0]["steps"][1:]; col = dict(M1="C0", M2="C1", knn10="C2", knn10y="C3", mlp="C4", null_knn_shuffled="gray", null_time_M2="k", null_time_knn10y="C5")
        for fi, fr in enumerate(ph[0]["frames"]):
            for m in KEYM:
                mi = ph[0]["models"].index(m); arr = np.stack([r["z"]["R2c_mean"][mi, fi] for r in ph if len(r["steps"]) == len(ph[0]["steps"])]); ax[fi].plot(st, np.nanmedian(arr, 0), c=col[m], label=m, ls=":" if m.startswith("null") else "-")
                if m == "M2": ax[fi].fill_between(st, np.nanpercentile(arr, 10, 0), np.nanpercentile(arr, 90, 0), color="C1", alpha=.15)
            ax[fi].set_xscale("log"); ax[fi].set_ylim(-1, 1.02); ax[fi].set_title(f"frame {fr}: held-out R2 (median over {len(ph)} phase runs; band = 10-90% of M2)"); ax[fi].set_xlabel("step"); ax[fi].axhline(0, c="k", lw=.4)
        ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{out}/figures/r2_vs_step.png", dpi=65); plt.close(fig)
    # --- heatmaps over (N, wd)
    Ns = sorted({r["N"] for r in ph}); wds = sorted({r["wd"] for r in ph})
    if ph:
        fig, ax = plt.subplots(3, 4, figsize=(16, 10))
        for fi, fr in enumerate(["raw", "procrustes", "affine"]):
            for mj, m in enumerate(["M1", "M2", "knn10y", "mlp"]):
                M = np.full((len(Ns), len(wds)), np.nan)
                for i, N in enumerate(Ns):
                    for j, wd in enumerate(wds):
                        v = [x[f"R2_{m}_{fr}_all10"] for x, rr in zip(rows, R) if rr["fam"] == "phase" and rr["probe"] == "test" and rr["N"] == N and rr["wd"] == wd]
                        if v: M[i, j] = np.nanmean(v)
                im = ax[fi, mj].imshow(M, origin="lower", cmap="viridis", aspect="auto", vmin=-0.1 if fr != "raw" else 0, vmax=1 if fr == "raw" else 0.6); ax[fi, mj].set_title(f"{fr}: {m} mean R2 (steps>=10)", fontsize=8)
                ax[fi, mj].set_xticks(range(len(wds))); ax[fi, mj].set_xticklabels([f"{w:g}" for w in wds], fontsize=6); ax[fi, mj].set_yticks(range(len(Ns))); ax[fi, mj].set_yticklabels(Ns, fontsize=6); plt.colorbar(im, ax=ax[fi, mj], fraction=.046)
                for i in range(len(Ns)):
                    for j in range(len(wds)):
                        if np.isfinite(M[i, j]): ax[fi, mj].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6, color="w")
        fig.tight_layout(); fig.savefig(f"{out}/figures/heatmap_R2.png", dpi=65); plt.close(fig)
    # --- attractor / eigenvalue time series for a few cells (affine frame), mean over seeds
    cells = [(250, 0.0), (1000, 0.0), (4000, 0.0), (4000, 1e-2), (60000, 0.0), (60000, 3e-2)]
    fig, ax = plt.subplots(3, 3, figsize=(16, 11)); ax = ax.ravel()
    for ci, (N, wd) in enumerate(cells):
        sel = [r for r in ph if r["N"] == N and r["wd"] == wd]
        if not sel: continue
        st = sel[0]["steps"]; s1 = st[1:]; dstep = np.diff(st).astype(float)
        for k, (key, lab, scale) in enumerate((("fp_dist_over_sep", "fixed-point distance to class mean / mean class sep", False), ("max_re", "max Re eig(A_y) per step", True), ("trace", "tr(A_y)/d... divergence per step", True))):
            for fr, ls in (("affine", "-"), ("raw", "--")):
                arr = np.stack([np.nanmedian(r["z"][f"char_{fr}_{key}"], 1) / (dstep if scale else 1) for r in sel]); ax[k * 3 + (ci % 3)].plot(s1, np.nanmean(arr, 0), ls=ls, label=f"N{N} wd{wd:g} {fr}")
            ax[k * 3 + (ci % 3)].set_xscale("log"); ax[k * 3 + (ci % 3)].set_title(lab, fontsize=8)
            if key != "fp_dist_over_sep": ax[k * 3 + (ci % 3)].set_yscale("symlog", linthresh=1e-6)
    for a_ in ax[:3]: a_.legend(fontsize=5)
    fig.tight_layout(); fig.savefig(f"{out}/figures/attractor_eigs.png", dpi=60); plt.close(fig)
    # --- mislabeled field
    gp = [r for r in R if r["fam"] == "gap" and r["probe"] == "train" and "mis_affine_cos_given" in r["z"].files]
    if gp:
        fig, ax = plt.subplots(2, 3, figsize=(17, 9)); cm = {250: "C0", 1000: "C1", 4000: "C2", 16000: "C3", 60000: "C4"}
        for fi, fr in enumerate(["raw", "procrustes", "affine"]):
            for r in gp:
                ep = r["steps"][1:] * 128.0 / r["N"]; d = r["z"][f"mis_{fr}_cos_given"] - r["z"][f"mis_{fr}_cos_true"]
                ax[0 if r["ln"] < .3 else 1, fi].plot(ep, d, c=cm[r["N"]], lw=.7, label=f"N{r['N']}" if r["seed"] == 0 else None)
            for k in (0, 1): ax[k, fi].axhline(0, c="k", lw=.5); ax[k, fi].set_xscale("log"); ax[k, fi].set_title(f"{fr}: cos(d,->given) - cos(d,->true) at mislabeled points, ln={0.2 if k==0 else 0.4}", fontsize=8); ax[k, fi].set_xlabel("epoch")
        ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{out}/figures/mislabeled_field.png", dpi=60); plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for ci, fr in enumerate(["affine", "procrustes"]):
            xs = [x[f"vf_cross_epoch_affine"] if fr == "affine" else x[f"vf_cross_step_procrustes"] * 128 / x["N"] for x in rows if "perclass_mem_cross_epoch" in x and np.isfinite(x.get("perclass_mem_cross_epoch", np.nan))]
            ys = [x["perclass_mem_cross_epoch"] for x in rows if "perclass_mem_cross_epoch" in x and np.isfinite(x.get("perclass_mem_cross_epoch", np.nan))]
            Ncol = [cm[x["N"]] for x in rows if "perclass_mem_cross_epoch" in x and np.isfinite(x.get("perclass_mem_cross_epoch", np.nan))]
            ax[ci].scatter(xs, ys, c=Ncol, s=14); lim = [1, 3e4]; ax[ci].plot(lim, lim, "k--", lw=.5); ax[ci].set_xscale("log"); ax[ci].set_yscale("log"); ax[ci].set_xlabel(f"vector-field crossover (epochs, {fr} frame)"); ax[ci].set_ylabel("memorization crossover (epochs, NCM persistent)")
        fig.tight_layout(); fig.savefig(f"{out}/figures/crossover_vs_memorization.png", dpi=60); plt.close(fig)
    print("done")


if __name__ == "__main__": main()
