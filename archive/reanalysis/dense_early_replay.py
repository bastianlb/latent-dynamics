#!/usr/bin/env python
"""Dense early-phase replay (CPU) of the probe/none D=16 baseline to test the 'under-sampled first logging
interval' objection: re-trains the same recipe (Adam lr 1e-3, batch 128, no wd) for the first N steps while
snapshotting the SAME probe set at steps 0,1,2,3,4,6,8,12,16,24,32,40,50,75,100,150,200 and computes the same
metrics (mean-pairwise-normalised). RNG stream differs from the original runs (it is a replay of the recipe, not of the
exact trajectory); progress is normalised by the ORIGINAL run's final value: p = (x_t - x_0)/(x_T - x_0).
Output: dense_early.csv, fig_dense_early.png   (standalone; only imports model.py / metrics.py read-only)
"""
from __future__ import annotations
import common as C

import csv, json, os
import numpy as np, torch, torch.nn as nn
from torchvision import datasets, transforms
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentdyn import model as MD
from latentdyn import metrics as M
from final_train_eval import tensors

torch.set_num_threads(2)
STEPS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 50, 75, 100, 150, 200]
KEYS = ["test_acc", "fisher_ratio", "h1_lifetime_sum_alpha1.0", "h1_entropy", "pmag", "raw_mean_norm"]


def run(ds, seed, D=16):
    cfg = json.load(open(os.path.join(C.RUNS, f"{ds}_probe_none_d{D}_s{seed}", "config.json")))
    MD.set_default_input_shape(cfg["in_channels"], cfg["img_size"]) if "in_channels" in cfg else MD.set_default_input_shape(3 if ds == "cifar10" else 1, 32 if ds == "cifar10" else 28)
    torch.manual_seed(seed); np.random.seed(seed)
    m = MD.LatentProbeModel(D, encoder_arch=cfg["encoder_arch"], objective="probe", latent_norm="none")
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xtr, ytr = tensors(ds, True); Xte, yte = tensors(ds, False)
    pidx = np.load(os.path.join(C.RUNS, f"{ds}_probe_none_d{D}_s{seed}", "probe_index.npz"))
    Xp, yp = Xte[pidx["indices"]], pidx["labels"]
    ce = nn.CrossEntropyLoss(); out = []; perm = torch.randperm(len(Xtr)); pos = 0
    for step in range(max(STEPS) + 1):
        if step in STEPS:
            m.eval()
            with torch.no_grad():
                z = m.encode(Xp).numpy(); acc = float(np.mean(np.concatenate([m(Xte[i:i + 2500])[0].argmax(1).numpy() for i in range(0, len(Xte), 2500)]) == yte.numpy()))
            met = M.compute_all_metrics(M.normalize_latents(z), yp)
            rawn = float(np.linalg.norm(z, axis=1).mean())
            out.append(dict(dataset=ds, seed=seed, step=step, test_acc=acc, raw_mean_norm=rawn, **{k: float(met[k]) for k in KEYS[1:5]}))
            m.train()
        if pos + 128 > len(Xtr): perm = torch.randperm(len(Xtr)); pos = 0
        b = perm[pos:pos + 128]; pos += 128
        logits, _ = m(Xtr[b]); loss = ce(logits, ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
    return out


def main():
    rows = []
    for ds in C.DATASETS:
        for s in (0, 1, 2):
            rows += run(ds, s); print(ds, s, "done", flush=True)
    with open(os.path.join(C.OUT, "dense_early.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    print("progress p=(x_t-x_0)/(x_T-x_0), x_T from the original 5-epoch runs; mean over 3 seeds; steps 0..200 (original first snapshot = step 50)")
    for ax, ds in zip(axs, C.DATASETS):
        for k in KEYS:
            xT = np.mean([C.load_timeseries(f"{ds}_probe_none_d16_s{s}")[k][-1] for s in (0, 1, 2)])
            P = []
            for s in (0, 1, 2):
                r = [x for x in rows if x["dataset"] == ds and x["seed"] == s]; v = np.array([x[k] for x in r]); P.append((v - v[0]) / (xT - v[0]))
            P = np.mean(P, 0); ax.plot(STEPS, P, "-o", ms=3, label=k)
            print(f"  {ds:13s} {k:26s} " + " ".join(f"{STEPS[i]}:{P[i]:+.2f}" for i in (1, 3, 5, 7, 9, 11, 12, 13, 14, 16)))
        ax.axvline(50, c="k", ls=":"); ax.set_xscale("symlog", linthresh=4); ax.set_title(f"{ds} dense early phase (replay)"); ax.set_ylim(-0.5, 1.6); ax.set_xlabel("step")
    axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "fig_dense_early.png"), dpi=110)


if __name__ == "__main__":
    main()
