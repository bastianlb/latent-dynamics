"""Subprocess launch helpers (generalises the loops in run_ae_*.py)."""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def train_cmd(*args) -> list:
    """[sys.executable, '-m', 'latentdyn.train', *args] (all args stringified)."""
    return [sys.executable, "-m", "latentdyn.train", *map(str, args)]


def run_jobs(cmds, env=None, workers=1, cwd=None, logs=None, skip=None, dry=False, names=None):
    """Run commands with bounded concurrency (threads around subprocess.call). Returns list of return codes
    (None for skipped/dry). cmds: list of argv lists. logs: optional parallel list of log-file paths
    (stdout+stderr redirected, else inherited). names: optional parallel list of run ids used in log lines. skip: optional predicate(i, cmd) -> bool. dry: print only."""
    def one(i):
        c = cmds[i]
        nm = names[i] if names is not None else None
        if skip is not None and skip(i, c):
            print("skip", nm if nm is not None else " ".join(map(str, c)), flush=True)
            return None
        if dry:
            print(" ".join(map(str, c)), flush=True)
            return None
        t = time.time()
        if logs is not None and logs[i] is not None:
            with open(logs[i], "w") as lf:
                rc = subprocess.call(c, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        else:
            rc = subprocess.call(c, cwd=cwd, env=env)
        print(f"{'ok' if rc == 0 else 'FAIL rc=%d' % rc}{' ' + str(nm) if nm is not None else ''} {time.time()-t:.0f}s", flush=True)
        return rc

    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(one, range(len(cmds))))
