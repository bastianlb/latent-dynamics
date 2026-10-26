"""Materialise the baseline tree (commit 71ee3eb) and import old modules by file path under unique names."""
import importlib.util
import os
import subprocess
import sys

BASE_COMMIT = "71ee3eb"
OLD_DIR = "/tmp/claude-2292/-homes-lbastian-latent-dynamics/a384d55c-229a-4c3c-8824-9becdaee1dc0/scratchpad/old_code"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def materialise():
    if not os.path.exists(os.path.join(OLD_DIR, "model.py")):
        os.makedirs(OLD_DIR, exist_ok=True)
        subprocess.run(f"git -C {REPO} archive {BASE_COMMIT} | tar -x -C {OLD_DIR}", shell=True, check=True)
    return OLD_DIR


def load_old(name):
    """Import OLD_DIR/<name>.py as module 'old_<name>' (only for modules without project-local imports)."""
    path = os.path.join(materialise(), name + ".py")
    spec = importlib.util.spec_from_file_location("old_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["old_" + name] = mod
    spec.loader.exec_module(mod)
    return mod
