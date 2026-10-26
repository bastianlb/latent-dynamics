"""Amendment 7 (review 5, post hoc) shared helpers. Import FIRST. CPU only, BLAS threads capped (<=8; step4_lib pins to 2).
step7_confirm.py is imported via step8_common, never modified."""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step8_common as H
import numpy as np
OUT6 = os.path.join(H.C.OUT_ROOT, "ext6"); os.makedirs(OUT6, exist_ok=True)
S, M, K7 = H.S, H.M, H.K7
CUTS, TARGETS = H.CUTS, H.TARGETS
CELLS = [(t, c) for t in TARGETS for c in CUTS]


def write_csv(fn, rows):
    import csv
    with open(f"{OUT6}/{fn}", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
