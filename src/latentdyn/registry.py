"""Reproducible registry of every training run under the artifacts root.

Scans run roots (directories whose `runs/<run_id>/config.json` exist), and writes
  results/runs.csv, results/config_defaults.json, results/sweeps.csv

Each run gets an exact `python -m latentdyn.train <flags>` command rebuilt from its
config.json via the train argparser, and VERIFIED by re-parsing it
(build_argparser -> Config -> resolve -> to_json_dict) and diffing with config.json.

Compared-away keys (IGNORED_KEYS): `device` (auto-detected; only `--device cpu` is
emitted, GPU runs say "cuda" because that is what the machine resolved at the time).
Saved-config keys that are absent from an older config.json but equal to the current
Config default are accepted (predates the flag); reported in `cmd_note`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from latentdyn.config import Config, DEFAULT_ARTIFACTS_ROOT, build_argparser

IGNORED_KEYS = ("device",)
# keys that are per-run identity/location, excluded from the "varying" column set
NON_VARYING = ("run_id", "artifacts_root")
FINAL_SCALARS = ("step", "epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                 "recon_train", "recon_test")
LEAD_COLS = ["dataset", "encoder_arch", "objective", "latent_norm", "latent_dim", "optimizer", "lr",
             "batch_size", "weight_decay", "seed", "epochs", "max_steps", "train_steps",
             "label_noise", "train_subset"]
# default set of roots (relative to the artifacts root) whose `runs/` subdir holds run dirs
DEFAULT_ROOTS = (".", "runs_v2", "ae_dynamics", "ae_dynamics/dense", "ae_dynamics/smoke",
                 "ae_dynamics/ntk", "ae_dynamics/ntk/lrctl", "dynamics_v2/dense",
                 "dynamics_v2/dense_latent/_run_artifacts", "dynamics_v2/dense/devtest/t1_dev",
                 "dynamics_v2/dense/devtest/t1_orig")

LAUNCHERS = {
    "ae_dynamics": "python scripts/ae/run_ae_grid.py",
    "ae_dynamics/dense": "python scripts/ae/run_ae_dense.py ; python scripts/ae/run_ae_dense_null.py",
    "ae_dynamics/ntk": "python scripts/ae/run_ae_ntk_ckpt.py",
    "dynamics_v2/dense": "(no launcher in repo; dense-optimizer-state runs)",
    "ae_dynamics/ntk/lrctl": "python scripts/ae/run_ae_lrctl.py",
}


# --------------------------------------------------------------------------- discovery
def discover_roots(artifacts_root: str, scan: List[str], depth: int = 3) -> List[str]:
    """Bounded walk (depth) below each `scan` dir for dirs named `runs` -> list of root rel paths."""
    skip = {"runs", "data", "latents", "latents_train", "figures", "metrics", "logs", "__pycache__",
            "phase_cache", "analysis"}
    found = set()
    A = Path(artifacts_root)

    def walk(p: Path, d: int, rel: str):
        try:
            ents = sorted(e for e in os.scandir(p) if e.is_dir(follow_symlinks=False))
        except OSError:
            return
        for e in ents:
            if e.name == "runs":
                found.add(rel)
            elif d > 0 and e.name not in skip:
                walk(Path(e.path), d - 1, f"{rel}/{e.name}" if rel != "." else e.name)
    for s in scan:
        walk(A / s if s != "." else A, depth, s)
        if (A / s / "runs").is_dir():
            found.add(s)
    return sorted(found)


# --------------------------------------------------------------------------- group maps
def jobs_blocks(artifacts_root: str) -> Dict[str, str]:
    """runs_v2: run_id -> block name, from runs_v2/jobs_<BLOCK>.txt (`<run_id>|<flags>` lines)."""
    m: Dict[str, str] = {}
    # HOLD files (jobs never launched as-is) sort last so a live jobs file wins
    for f in sorted(glob.glob(f"{artifacts_root}/runs_v2/jobs_*.txt"), key=lambda f: ("HOLD" in f, f)):
        block = os.path.basename(f)[len("jobs_"):-len(".txt")]
        with open(f) as fh:
            for line in fh:
                if "|" in line:
                    m.setdefault(line.split("|", 1)[0].strip(), block)
    return m


def sweep_manifests(artifacts_root: str) -> Dict[str, str]:
    """top-level runs/: run_id -> sweep_name, from sweeps/*/manifest.json."""
    m: Dict[str, str] = {}
    for f in sorted(glob.glob(f"{artifacts_root}/sweeps/*/manifest.json")):
        try:
            man = json.load(open(f))
        except Exception:
            continue
        runs = man.get("runs", {})
        ids = runs.keys() if isinstance(runs, dict) else [r.get("run_id") for r in runs]
        for rid in ids:
            m.setdefault(rid, man.get("sweep_name", os.path.basename(os.path.dirname(f))))
    return m


# --------------------------------------------------------------------------- command build/verify
_PARSER = build_argparser()
_ACTIONS = [a for a in _PARSER._actions if a.option_strings and a.dest not in ("help", "smoke")]


def config_to_flags(cfg: dict) -> List[str]:
    """Flags (as argv list) that reproduce `cfg`: every value differing from the parser default."""
    argv: List[str] = []
    for a in _ACTIONS:
        d = a.dest
        if d == "device":
            if cfg.get("device") == "cpu":
                argv += ["--device", "cpu"]
            continue
        if d == "run_id":
            continue  # handled below
        if d not in cfg:
            continue
        v = cfg[d]
        flag = a.option_strings[0]
        kind = type(a).__name__
        if kind == "_StoreTrueAction":
            if v:
                argv.append(flag)
        elif kind == "_StoreFalseAction":
            if v is False:
                argv.append(flag)
        elif d == "artifacts_root":
            argv += [flag, str(v)]  # always explicit
        else:
            if v != a.default and v is not None:
                argv += [flag, repr(v) if isinstance(v, float) else str(v)]
    return argv


def _parse(argv: List[str]) -> dict:
    ns = _PARSER.parse_args(argv)
    kw = vars(ns).copy()
    kw.pop("smoke")
    return Config(**kw).resolve().to_json_dict()


def build_and_verify(cfg: dict):
    """-> (argv, verified, mismatch_keys, notes). Adds --run_id only if derivation differs."""
    argv = config_to_flags(cfg)
    try:
        parsed = _parse(argv)
        if parsed.get("run_id") != cfg.get("run_id"):
            argv += ["--run_id", str(cfg.get("run_id"))]
            parsed = _parse(argv)
    except SystemExit:
        return argv, False, ["<parse_error>"], []
    except Exception as e:  # noqa: BLE001
        return argv, False, [f"<{type(e).__name__}>"], []
    defaults = Config().to_json_dict()
    mism, notes = [], []
    for k in sorted(set(parsed) | set(cfg)):
        if k in IGNORED_KEYS:
            continue
        if k not in cfg:
            if parsed[k] == defaults.get(k, object()) or k not in defaults:
                notes.append(f"absent:{k}")
            else:
                mism.append(f"missing:{k}")
        elif k not in parsed:
            mism.append(f"extra:{k}")
        elif parsed[k] != cfg[k]:
            mism.append(k)
    return argv, not mism, mism, notes


# --------------------------------------------------------------------------- per-run read
def _final_snapshot(run_dir: str) -> dict:
    out = {"final_step": None}
    try:
        names = [n for n in os.listdir(f"{run_dir}/latents") if n.startswith("step_") and n.endswith(".npz")]
        if not names:
            return out
        last = max(names, key=lambda n: int(n[5:-4]))
        with np.load(f"{run_dir}/latents/{last}") as z:  # lazy: only scalar keys touched
            for k in FINAL_SCALARS:
                if k in z.files:
                    out[k if k != "step" else "final_step"] = float(z[k]) if k != "step" else int(z[k])
    except Exception as e:  # noqa: BLE001
        out["read_error"] = type(e).__name__
    return out


def _load_run(args):
    artifacts_root, root_rel, run_id = args
    root = artifacts_root if root_rel == "." else f"{artifacts_root}/{root_rel}"
    d = f"{root}/runs/{run_id}"
    try:
        cfg = json.load(open(f"{d}/config.json"))
    except (OSError, ValueError):
        return None
    rel = os.path.relpath(d, artifacts_root)
    rec = {"root": root_rel, "run_id": run_id, "rel_path": rel, "cfg": cfg,
           "status": "complete" if os.path.exists(f"{d}/model_final.pt") else "incomplete"}
    rec.update(_final_snapshot(d))
    return rec


# --------------------------------------------------------------------------- build
def _group(rec, blocks, sweeps) -> str:
    r = rec["root"]
    if r == ".":
        s = sweeps.get(rec["run_id"])
        return f"runs/{s}" if s else "runs"
    if r == "runs_v2":
        rid = rec["run_id"]
        if rid in blocks:
            return f"runs_v2/{blocks[rid]}"
        # not in any jobs_*.txt: launched by the standalone *_block.sh scripts (identified by run_id tag)
        for pat, name in (("_phase_", "mnist_block_A"), ("_gap_", "mnist_block_B"),
                          ("_rn18_", "cifar_block"), ("_l2t_", "l2t_block")):
            if pat in rid:
                return f"runs_v2/{name}"
        return "runs_v2/unlisted"
    return r


def build(artifacts_root: str, out_dir: str, roots: Optional[List[str]] = None, threads: int = 8):
    roots = list(roots) if roots else list(DEFAULT_ROOTS)
    tasks = []
    for r in roots:
        base = artifacts_root if r == "." else f"{artifacts_root}/{r}"
        try:
            ids = sorted(os.listdir(f"{base}/runs"))
        except OSError:
            continue
        tasks += [(artifacts_root, r, i) for i in ids]
    with ThreadPoolExecutor(threads) as ex:
        recs = [x for x in ex.map(_load_run, tasks) if x]
    blocks, sweeps = jobs_blocks(artifacts_root), sweep_manifests(artifacts_root)
    for rec in recs:
        rec["group"] = _group(rec, blocks, sweeps)
        argv, ok, mism, notes = build_and_verify(rec["cfg"])
        rec["command"] = "python -m latentdyn.train " + shlex.join(argv)
        rec["cmd_verified"], rec["cmd_mismatch"], rec["cmd_note"] = ok, ";".join(mism), ";".join(notes)
    recs.sort(key=lambda r: (r["group"], r["run_id"]))

    # varying vs constant config keys
    keys = sorted({k for r in recs for k in r["cfg"]})
    vals = {k: {json.dumps(r["cfg"].get(k, "<absent>"), sort_keys=True) for r in recs} for k in keys}
    const = {k: json.loads(next(iter(v))) for k, v in vals.items() if len(v) == 1 and k not in NON_VARYING}
    varying = [k for k in keys if len(vals[k]) > 1 and k not in NON_VARYING]
    lead = [k for k in LEAD_COLS if k in keys]  # always shown even if constant
    extra = [k for k in varying if k not in lead]
    cols = (["group", "run_id", "rel_path"] + lead + extra +
            ["status", "final_step", "epoch", "train_loss", "train_acc", "test_loss", "test_acc",
             "recon_train", "recon_test", "cmd_verified", "cmd_mismatch", "cmd_note", "command"])
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            row = {c: r.get(c) for c in cols}
            for k in lead + extra:
                v = r["cfg"].get(k)
                row[k] = "" if v is None else v
            w.writerow(row)
    with open(f"{out_dir}/config_defaults.json", "w") as fh:
        json.dump({"note": "config keys constant across ALL runs in runs.csv; run_id/artifacts_root excluded",
                   "ignored_in_verification": list(IGNORED_KEYS), "constants": const}, fh, indent=1, sort_keys=True)
    _write_sweeps(recs, extra + [k for k in lead], f"{out_dir}/sweeps.csv")
    return recs, roots


def _sweep_cmd(group: str, rs: list) -> str:
    if group in LAUNCHERS:
        return LAUNCHERS[group]
    BS = {"mnist_block_A": "mnist_block.sh", "mnist_block_B": "mnist_block.sh",
          "cifar_block": "cifar_block.sh <gpu_index>", "l2t_block": "l2t_block.sh"}
    if group.startswith("runs_v2/") and group.split("/", 1)[1] in BS:
        return "bash runs_v2/" + BS[group.split("/", 1)[1]]
    if group.startswith("runs_v2/"):
        return f"bash runs_v2/run_queue.sh runs_v2/jobs_{group.split('/', 1)[1]}.txt"
    if group.startswith("runs/"):
        c = [r["cfg"] for r in rs]
        u = lambda k: sorted({x[k] for x in c})  # noqa: E731
        s = lambda k: " ".join(map(str, u(k)))  # noqa: E731
        return (f"python -m latentdyn.sweep --sweep_name {group.split('/', 1)[1]} --latent_dims {s('latent_dim')} "
                f"--seeds {s('seed')} --objective {s('objective')} --latent_norm {s('latent_norm')} "
                f"--epochs {u('epochs')[0]} --log_every {u('log_every')[0]} --dense_early {u('dense_early')[0]}"
                "  # approximate (pair structure may use --variant); exact per-run commands in runs.csv")
    return ""


def _write_sweeps(recs, cand_keys, path):
    by = defaultdict(list)
    for r in recs:
        by[r["group"]].append(r)
    rows = []
    for g, rs in sorted(by.items()):
        var = {}
        for k in sorted(set(cand_keys)):
            vs = {json.dumps(r["cfg"].get(k), sort_keys=True) for r in rs}
            if len(vs) > 1 and k not in NON_VARYING and k != "tag":
                lv = [json.loads(v) for v in vs]
                var[k] = "..." if any(isinstance(x, (list, dict)) for x in lv) else sorted(lv, key=lambda x: (str(type(x)), str(x)) if not isinstance(x, (int, float)) or isinstance(x, bool) else ("0", x))[:12]
        rows.append({"group": g, "n_runs": len(rs), "n_complete": sum(r["status"] == "complete" for r in rs),
                     "n_cmd_verified": sum(bool(r["cmd_verified"]) for r in rs),
                     "varying": json.dumps(var, sort_keys=True, default=str), "launcher": _sweep_cmd(g, rs)})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["group", "n_runs", "n_complete", "n_cmd_verified", "varying", "launcher"])
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts_root", default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "results"))
    ap.add_argument("--roots", nargs="*", default=None, help="run roots rel. to artifacts root ('.' = top level)")
    ap.add_argument("--discover", nargs="*", default=None, help="scan these dirs (depth 3) for extra `runs/` roots")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    roots = a.roots or list(DEFAULT_ROOTS)
    if a.discover:
        roots = sorted(set(roots) | set(discover_roots(a.artifacts_root, a.discover)))
    recs, roots = build(a.artifacts_root, a.out, roots, a.threads)
    print("roots:", roots)
    g = Counter(r["group"] for r in recs)
    for k, v in sorted(g.items()):
        print(f"  {k}: {v}")
    ok = sum(bool(r["cmd_verified"]) for r in recs)
    print(f"runs={len(recs)} complete={sum(r['status']=='complete' for r in recs)} verified={ok}/{len(recs)}")
    h = Counter(k for r in recs for k in r["cmd_mismatch"].split(";") if k)
    print("mismatch histogram:", dict(h.most_common()))
    n = Counter(k for r in recs for k in r["cmd_note"].split(";") if k)
    print("absent-but-default notes:", dict(n.most_common()))


if __name__ == "__main__":
    main()
