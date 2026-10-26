"""Thread-pool caps for numpy/BLAS/ripser. Must run BEFORE numpy is imported. No numpy/torch import here."""
import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def cap_threads(n=8, force=False):
    """Cap BLAS/OpenMP thread pools to n. force=False keeps user-set values (setdefault); force=True overrides."""
    for k in THREAD_VARS:
        if force:
            os.environ[k] = str(n)
        else:
            os.environ.setdefault(k, str(n))
