#!/usr/bin/env python
"""Assemble caches into tidy CSVs: metrics_long.csv (run_id,probe_set,step,metric,value) and runs.csv (config table)."""
import common2 as C
import data2 as D
import csv, numpy as np

runs = D.load_runs()
rows_cfg = []
with open(f"{C.OUT}/metrics_long.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run_id", "probe_set", "step", "metric", "value"])
    for r in runs:
        d, rid, st = r["d"], r["info"]["run_id"], r["step"]
        for k in d.files if hasattr(d, "files") else d:
            if k.startswith("run/"):
                ps, m = "run", k[4:]
            elif k.split("/")[0] in C.PROBES and k.split("/")[1] != "mag":
                ps, m = k.split("/")
            else:
                continue
            v = d[k]
            for i in np.where(np.isfinite(v))[0]:
                w.writerow([rid, ps, int(st[i]), m, f"{v[i]:.6g}"])
        w.writerow([rid, "run", 0, "gap_def", "train_acc-test_acc"]) if False else None
        gap = r["gap"]
        for i in range(len(st)):
            w.writerow([rid, "run", int(st[i]), "gap", f"{gap[i]:.6g}"])
        for ps in C.PROBES:
            mg = d[f"{ps}/mag"]
            for i in np.where(d["topo_mask"])[0]:
                for j, t in enumerate(C.MAG_GRID):
                    w.writerow([rid, ps, int(st[i]), f"mag_t{t:.3g}", f"{mg[i, j]:.6g}"])
        rows_cfg.append(dict(**r["info"], slingshot_onset=r["onset"], pre_step=int(st[r["pre"]]), n_snap=len(st),
                             peak_test_acc=r["peak_acc"], peak_step=r["peak_step"], cross_step=r["cross_step"]))
keys = list(rows_cfg[0].keys())
with open(f"{C.OUT}/runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows_cfg)
print(len(runs), "runs written")
