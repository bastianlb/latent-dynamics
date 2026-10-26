"""Old (baseline 71ee3eb) vs new pipeline: analyze/trajectory/plot/compare on a synthetic run, CPU only."""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _oldcode  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
           MPLBACKEND="Agg")


def _run(args, cwd):
    subprocess.run([sys.executable, *args], cwd=cwd, env=ENV, check=True, capture_output=True)


def _make_runs(tmp):
    from latentdyn import metrics as M
    a = M._make_synthetic_run(os.path.join(tmp, "old", "run"), n_steps=6, seed=1)
    b = os.path.join(tmp, "new", "run")
    shutil.copytree(a, b)
    return a, b


def _cmp_npz(p, q):
    A, B = np.load(p), np.load(q)
    assert sorted(A.files) == sorted(B.files)
    for k in A.files:
        assert np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind == "f"), k


def test_pipeline_equiv():
    old = _oldcode.materialise()
    tmp = tempfile.mkdtemp(prefix="equiv_pipeline_")
    try:
        ro, rn = _make_runs(tmp)
        _run(["analyze.py", "--run_dir", ro], old)
        _run(["-m", "latentdyn.pipeline.analyze", "--run_dir", rn], REPO)
        _cmp_npz(os.path.join(ro, "metrics", "timeseries.npz"), os.path.join(rn, "metrics", "timeseries.npz"))
        assert open(os.path.join(ro, "metrics", "timeseries.csv")).read() == \
            open(os.path.join(rn, "metrics", "timeseries.csv")).read()
        _run(["plot.py", "--run_dir", ro], old)
        _run(["-m", "latentdyn.pipeline.plot", "--run_dir", rn], REPO)
        _run(["trajectory.py", "--run_dir", ro, "--n_frames", "4", "--grid_steps", "4"], old)
        _run(["-m", "latentdyn.pipeline.trajectory", "--run_dir", rn, "--n_frames", "4", "--grid_steps", "4"], REPO)
        _run(["compare.py", "--run_dirs", ro, "--group_by", "latent_dim", "--out_dir", os.path.join(tmp, "co")], old)
        _run(["-m", "latentdyn.pipeline.compare", "--run_dirs", rn, "--group_by", "latent_dim",
              "--out_dir", os.path.join(tmp, "cn")], REPO)
        fo = sorted(os.path.relpath(p, ro) for p in glob.glob(ro + "/**/*.png", recursive=True))
        fn = sorted(os.path.relpath(p, rn) for p in glob.glob(rn + "/**/*.png", recursive=True))
        assert fo == fn and fo, (fo, fn)
        assert sorted(os.listdir(os.path.join(tmp, "co"))) == sorted(os.listdir(os.path.join(tmp, "cn")))
        print("pngs:", len(fo))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline_equiv()
    print("PASS")
