#!/usr/bin/env python
"""Offline linear probe on FULL train/test latents of each run's final checkpoint (calibration for the
1000-point CV probe): fit standardised LogisticRegression on encoder latents of the full training set,
report test accuracy. Output: <OUT>/final_offline_probe_full.csv"""
from __future__ import annotations
import common as C
import csv, json, os, warnings
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import model as MD
from final_train_eval import tensors

torch.set_num_threads(2)


@torch.no_grad()
def encode(m, X):
    m.eval(); return torch.cat([m.encode(X[i:i + 2000]) for i in range(0, len(X), 2000)]).numpy()


def main():
    warnings.filterwarnings("ignore"); rows = []
    for r in C.list_runs():
        cfg = json.load(open(os.path.join(C.RUNS, r["run_id"], "config.json")))
        MD.set_default_input_shape(cfg.get("in_channels", 3 if r["dataset"] == "cifar10" else 1), cfg.get("img_size", 32 if r["dataset"] == "cifar10" else 28))
        m = MD.LatentProbeModel(cfg["latent_dim"], encoder_arch=cfg["encoder_arch"], objective=cfg["objective"], latent_norm=cfg["latent_norm"])
        m.load_state_dict(torch.load(os.path.join(C.RUNS, r["run_id"], "model_final.pt"), map_location="cpu"))
        Xtr, ytr = tensors(r["dataset"], True); Xte, yte = tensors(r["dataset"], False)
        Ztr, Zte = encode(m, Xtr), encode(m, Xte)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Ztr, ytr.numpy())
        rows.append(dict(run_id=r["run_id"], offline_full_test_acc=clf.score(Zte, yte.numpy()), offline_full_train_acc=clf.score(Ztr, ytr.numpy())))
        print(rows[-1], flush=True)
    with open(os.path.join(C.OUT, "final_offline_probe_full.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
