#!/usr/bin/env python
"""Capacity-sweep saturation table (probe/none, D in 2..32) per dataset: test acc, Fisher, H1_L1, PMag, saturation D
(smallest D within 1 / 2 accuracy points of D=32), Fisher argmax D.  Output: capacity_saturation.csv"""
from __future__ import annotations
import common as C
import csv, os
import numpy as np


def main():
    rows = []
    for ds in C.DATASETS:
        acc = {}
        for D in (2, 4, 8, 16, 32):
            rr = [r for r in C.list_runs([ds]) if r["objective"] == "probe" and r["norm"] == "none" and r["D"] == D]
            ts = [C.load_timeseries(r["run_id"]) for r in rr]
            g = lambda k: np.array([t[k][-1] for t in ts])
            ca = [C.load_cache(r["run_id"]) for r in rr]
            off = np.mean([c["offline_probe_acc"][-1] for c in ca]) if all(c is not None for c in ca) else np.nan
            acc[D] = g("test_acc").mean()
            rows.append(dict(dataset=ds, D=D, test_acc=g("test_acc").mean(), test_acc_sd=g("test_acc").std(ddof=1), offline_cv_acc=off,
                             fisher=g("fisher_ratio").mean(), h1_L1=g("h1_lifetime_sum_alpha1.0").mean(), pmag=g("pmag").mean()))
        sat1 = min(D for D in acc if acc[D] >= acc[32] - 0.01); sat2 = min(D for D in acc if acc[D] >= acc[32] - 0.02)
        fD = max((r for r in rows if r["dataset"] == ds), key=lambda r: r["fisher"])["D"]
        print(f"{ds}: acc by D = " + ", ".join(f"D{D}:{acc[D]:.3f}" for D in acc) + f" | saturation D (within 1pt of D=32): {sat1}; within 2pt: {sat2}; Fisher argmax D = {fD}")
        for r in rows:
            if r["dataset"] == ds: r["sat_D_1pt"] = sat1; r["sat_D_2pt"] = sat2; r["fisher_argmax_D"] = fD
    with open(os.path.join(C.OUT, "capacity_saturation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
