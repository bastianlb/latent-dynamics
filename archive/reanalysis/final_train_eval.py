#!/usr/bin/env python
"""Eval-mode FULL train-set and test-set accuracy of each run's final checkpoint (model_final.pt).

Why: the snapshot `train_acc` of the original runs is the *running* mean over the training
batches since the previous snapshot (train-mode, moving weights), so train_acc - test_acc
is not a clean generalisation gap. This recomputes a proper end-of-training gap on CPU.
Standalone (only imports model.py read-only; does not touch train.py/data.py).
Output: <OUT>/final_train_eval.csv  (run_id, train_acc_eval, test_acc_eval, gap_eval)
"""
from __future__ import annotations
import common as C

import csv, json, os, sys
import numpy as np
import torch
from torchvision import datasets, transforms

from latentdyn import model as MD

torch.set_num_threads(2)
NORM = {"mnist": ((0.1307,), (0.3081,)), "fashion_mnist": ((0.2860,), (0.3530,)),
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}
CLS = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}
_cache = {}


def tensors(ds, train):
    if (ds, train) not in _cache:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORM[ds])])
        d = CLS[ds](root=os.path.join(C.ART, "data"), train=train, download=False, transform=tf)
        X = torch.stack([d[i][0] for i in range(len(d))]); y = torch.tensor(d.targets)
        _cache[(ds, train)] = (X, y)
    return _cache[(ds, train)]


@torch.no_grad()
def acc(m, X, y):
    m.eval(); c = 0
    for i in range(0, len(X), 1000):
        logits, _ = m(X[i:i + 1000]); c += int((logits.argmax(1) == y[i:i + 1000]).sum())
    return c / len(X)


def main():
    out = os.path.join(C.OUT, "final_train_eval.csv")
    rows = []
    for r in C.list_runs():
        cfg = json.load(open(os.path.join(C.RUNS, r["run_id"], "config.json")))
        MD.set_default_input_shape(cfg["in_channels"] if "in_channels" in cfg else (3 if r["dataset"] == "cifar10" else 1),
                                   cfg.get("img_size", 32 if r["dataset"] == "cifar10" else 28))
        m = MD.LatentProbeModel(cfg["latent_dim"], encoder_arch=cfg["encoder_arch"], objective=cfg["objective"],
                                latent_norm=cfg["latent_norm"])
        sd = torch.load(os.path.join(C.RUNS, r["run_id"], "model_final.pt"), map_location="cpu")
        m.load_state_dict(sd)
        Xtr, ytr = tensors(r["dataset"], True); Xte, yte = tensors(r["dataset"], False)
        tr, te = acc(m, Xtr, ytr), acc(m, Xte, yte)
        ts = C.load_timeseries(r["run_id"])
        rows.append(dict(run_id=r["run_id"], train_acc_eval=tr, test_acc_eval=te, gap_eval=tr - te,
                         snap_train_acc=ts["train_acc"][-1], snap_test_acc=ts["test_acc"][-1],
                         snap_gap=ts["train_acc"][-1] - ts["test_acc"][-1]))
        print(rows[-1], flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
