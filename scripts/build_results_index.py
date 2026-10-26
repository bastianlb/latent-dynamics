"""Build results/runs.csv, results/config_defaults.json, results/sweeps.csv (see latentdyn.registry)."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from latentdyn.registry import main

if __name__ == "__main__":
    main()
