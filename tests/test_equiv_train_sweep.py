"""Equivalence of latentdyn.sweep / latentdyn.train vs the baseline (71ee3eb) run_sweep.py / train.py.
CPU only. Old code runs in subprocesses from the materialised baseline tree."""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _oldcode import materialise  # noqa: E402

PY = sys.executable
SCRATCH = "/tmp/claude-2292/-homes-lbastian-latent-dynamics/a384d55c-229a-4c3c-8824-9becdaee1dc0/scratchpad"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")

DRIVER = r'''
import json, subprocess, sys, runpy
cmds = []
class P:
    pid = 1
    def __init__(self, cmd, **kw): cmds.append(list(cmd))
    def poll(self): return 0
    def terminate(self): pass
subprocess.Popen = P
sys.argv = ["x"] + json.loads(sys.argv[1])
{call}
print("CMDS=" + json.dumps(cmds))
'''
OLD_CALL = "import run_sweep; run_sweep.main()"
NEW_CALL = "from latentdyn import sweep; sweep.main()"

SWEEPS = [
    ["--sweep_name", "t1", "--latent_dims", "2", "4", "--seeds", "0", "1", "--no_gpu_check", "--poll_interval", "0"],
    ["--sweep_name", "t2", "--variant", "probe:none", "--variant", "autoencoder:l2", "--latent_dims", "8",
     "--tag", "x", "--device", "cpu", "--max_steps", "5", "--dense_early", "3", "--no_gpu_check", "--poll_interval", "0"],
    ["--sweep_name", "t3", "--smoke", "--latent_dims", "2", "--no_gpu_check", "--poll_interval", "0"],
]


def _cmds(call, cwd, argv, root):
    code = DRIVER.replace("{call}", call)
    r = subprocess.run([PY, "-c", code, json.dumps(argv + ["--artifacts_root", root])], cwd=cwd, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("CMDS=")][0]
    return json.loads(line[5:])


def test_sweep_cmds():
    old = materialise()
    for argv in SWEEPS:
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            o = _cmds(OLD_CALL, old, argv, root)
            n = _cmds(NEW_CALL, os.path.dirname(old), argv, root)
        assert len(o) == len(n) > 0
        for a, b in zip(o, n):
            assert a[1].endswith("train.py") and b[1:3] == ["-m", "latentdyn.train"], (a[:2], b[:3])
            assert a[2:] == b[3:], (a, b)


def test_train_bitwise():
    import numpy as np
    old = materialise()
    data = "/vol/bitbucket/lbastian/latent-dynamics/data"
    outs = {}
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        for name in ("old", "new"):
            root = os.path.join(tmp, name)
            os.makedirs(root)
            os.symlink(data, os.path.join(root, "data"))
            args = ["--smoke", "--device", "cpu", "--max_steps", "6", "--seed", "3", "--artifacts_root", root]
            cmd = [PY, "train.py"] + args if name == "old" else [PY, "-m", "latentdyn.train"] + args
            r = subprocess.run(cmd, cwd=old if name == "old" else tmp, env=ENV, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            files = {}
            for d, _, fs in os.walk(os.path.join(root, "runs")):
                for f in fs:
                    if f.endswith(".npz"):
                        files[os.path.relpath(os.path.join(d, f), root)] = dict(np.load(os.path.join(d, f)))
            outs[name] = files
        assert outs["old"].keys() == outs["new"].keys() and outs["old"], outs["old"].keys()
        for k in outs["old"]:
            assert outs["old"][k].keys() == outs["new"][k].keys()
            for kk in outs["old"][k]:
                assert np.array_equal(outs["old"][k][kk], outs["new"][k][kk], equal_nan=True), (k, kk)


if __name__ == "__main__":
    test_sweep_cmds(); print("sweep cmds OK")
    test_train_bitwise(); print("train bitwise OK")
