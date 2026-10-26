"""L2-temperature study summary: Fisher ratio, linear probe, accuracies, learned scale.

Reads runs_v2 *_l2t_* runs; writes CSV + bar plot to the bitbucket analysis dir.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import re
import glob
import csv

import numpy as np

sys.path.insert(0, "/homes/lbastian/latent-dynamics")
from latentdyn.metrics import fisher_separation_ratio, offline_linear_separability  # noqa: E402

RUNS = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/l2t"
NAME_RE = re.compile(r"^(mnist|fashion_mnist)_probe_(none|l2)_d16_s(\d+)_l2t_(\w+)$")
TARGETS = {"final": None, "step500": 500}
VARIANT_ORDER = ["none", "s1", "s10", "learn10"]


def load_snap(path):
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    return out


def nearest_step_file(folder, target):
    files = sorted(glob.glob(os.path.join(folder, "step_*.npz")))
    steps = [(int(re.search(r"step_(\d+)\.npz", f).group(1)), f) for f in files]
    if target is None:
        return max(steps)
    return min(steps, key=lambda sf: abs(sf[0] - target))


def train_snap_for_step(run_dir, step):
    tdir = os.path.join(run_dir, "latents_train")
    fpath = os.path.join(tdir, f"step_{step:07d}.npz")
    if os.path.exists(fpath):
        return load_snap(fpath)
    s, f = nearest_step_file(tdir, step)
    return load_snap(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    run_dirs = sorted(glob.glob(os.path.join(RUNS, "*_l2t_*")))
    for run_dir in run_dirs:
        m = NAME_RE.match(os.path.basename(run_dir))
        if not m:
            continue
        dataset, _, seed, tag = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        variant = tag
        final_step_file = nearest_step_file(os.path.join(run_dir, "latents"), None)
        final_step = final_step_file[0]
        for tname, target in TARGETS.items():
            s_test_step, f_test = nearest_step_file(os.path.join(run_dir, "latents"),
                                                    target if target is not None else None)
            test = load_snap(f_test)
            train = train_snap_for_step(run_dir, int(test["step"]))
            fisher_test = fisher_separation_ratio(test["latents"], test["labels"])
            fisher_train = fisher_separation_ratio(train["latents"], train["labels"])
            lin_acc = offline_linear_separability(test["latents"], test["labels"])
            l2 = float(test["l2_scale"]) if "l2_scale" in test else float("nan")
            rows.append({
                "dataset": dataset, "variant": variant, "seed": seed, "target": tname,
                "step": int(test["step"]), "fisher_test": fisher_test,
                "fisher_train": fisher_train, "offline_acc_test": lin_acc,
                "test_acc": float(test["test_acc"]), "train_acc": float(test["train_acc"]),
                "l2_scale": l2, "run": os.path.basename(run_dir),
            })
            print(dataset, variant, seed, tname, int(test["step"]),
                  f"F_test={fisher_test:.4f} F_train={fisher_train:.4f} lin={lin_acc:.4f}", flush=True)

    # per-run CSV
    fields = list(rows[0].keys())
    with open(os.path.join(OUT, "l2t_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # aggregate per (dataset, variant, target)
    metrics = ["fisher_test", "fisher_train", "offline_acc_test", "test_acc", "train_acc", "l2_scale"]
    agg = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["variant"], r["target"]), []).append(r)

    summary = []
    for (ds, var, tgt), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], VARIANT_ORDER.index(kv[0][1]), kv[0][2])):
        row = {"dataset": ds, "variant": var, "target": tgt, "n_seeds": len(rs)}
        for mname in metrics:
            vals = np.array([r[mname] for r in rs], dtype=float)
            row[f"{mname}_mean"] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
            row[f"{mname}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 and np.any(~np.isnan(vals)) else float("nan")
        summary.append(row)

    # ratio vs none, paired by seed, final step
    ratio_rows = []
    for ds in ["mnist", "fashion_mnist"]:
        none_by_seed = {r["seed"]: r["fisher_test"] for r in rows
                        if r["dataset"] == ds and r["variant"] == "none" and r["target"] == "final"}
        for var in ["s1", "s10", "learn10"]:
            ratios = []
            for r in rows:
                if r["dataset"] == ds and r["variant"] == var and r["target"] == "final":
                    ratios.append(r["fisher_test"] / none_by_seed[r["seed"]])
            ratios = np.array(ratios)
            ratio_rows.append({
                "dataset": ds, "variant": var, "target": "final",
                "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std(ddof=1)),
                "per_seed": " ".join(f"{x:.3f}" for x in ratios),
            })

    sum_fields = list(summary[0].keys())
    with open(os.path.join(OUT, "l2t_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(OUT, "l2t_ratio_vs_none.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ratio_rows[0].keys()))
        w.writeheader()
        w.writerows(ratio_rows)

    # plot: final-step Fisher ratio, grouped by dataset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mnist": "#2b6cb0", "fashion_mnist": "#dd6b20"}
    variants = VARIANT_ORDER
    x = np.arange(len(variants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, ds in enumerate(["mnist", "fashion_mnist"]):
        means, stds = [], []
        for var in variants:
            srow = next(s for s in summary if s["dataset"] == ds and s["variant"] == var and s["target"] == "final")
            means.append(srow["fisher_test_mean"])
            stds.append(srow["fisher_test_std"])
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4,
               color=colors[ds], label=ds, error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(["none", "s1", "s10", "learn10"])
    ax.set_xlabel("head variant (l2 temperature)")
    ax.set_ylabel("Fisher ratio (test probe, final step)")
    ax.set_title("Fisher separation ratio by variant (D=16, mean ± std over 3 seeds)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "l2t_fisher.png"), dpi=150)

    print("\nSUMMARY")
    for s in summary:
        print(s["dataset"], s["variant"], s["target"],
              f"fisher={s['fisher_test_mean']:.3f}±{s['fisher_test_std']:.3f}",
              f"lin={s['offline_acc_test_mean']:.4f}±{s['offline_acc_test_std']:.4f}",
              f"test={s['test_acc_mean']:.4f}±{s['test_acc_std']:.4f}",
              f"scale={s['l2_scale_mean']:.3f}±{s['l2_scale_std']:.3f}")
    print("\nRATIO vs none (final, paired by seed)")
    for r in ratio_rows:
        print(r)


if __name__ == "__main__":
    main()
