"""Runs metrics' built-in end-to-end self test (needs latentdyn.pipeline.analyze/plot, CPU only)."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
from latentdyn.metrics import _run_self_test


def test_metrics_selftest():
    assert _run_self_test()


if __name__ == "__main__":
    test_metrics_selftest()
