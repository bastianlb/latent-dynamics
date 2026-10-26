"""Launch a grid of `latentdyn.train` runs as co-located subprocesses on whatever
GPU(s) the COORDINATOR has already pinned.

    ****************************************************************
    * latentdyn.sweep NEVER chooses or sets CUDA_VISIBLE_DEVICES.      *
    * It is a pure orchestrator: it inherits the environment it is  *
    * launched with (via `env=os.environ.copy()` for every child)  *
    * and never touches the GPU itself (no `import torch`, no      *
    * `nvidia-smi` calls that *select* a device -- only read-only   *
    * memory queries restricted to the indices already listed in   *
    * CUDA_VISIBLE_DEVICES, used purely as an optional wait-gate).  *
    * The CALLER (coordinator) must `export CUDA_VISIBLE_DEVICES=`  *
    * to 1-2 idle GPU indices, per CLAUDE.md, BEFORE invoking this  *
    * script. Every child subprocess inherits that same env var     *
    * unchanged and lets train.py's own device auto-detection pick  *
    * it up (or pass --device explicitly to force cpu/cuda).        *
    ****************************************************************

Grid construction:
  - The full grid is the cartesian product of
      (objective, latent_norm) pairs  x  latent_dims  x  seeds
  - By default the (objective, latent_norm) pairs are the cartesian product
    of --objective x --latent_norm (each may repeat for multiple values).
  - Pass one or more --variant OBJECTIVE:LATENT_NORM to instead give an
    EXPLICIT list of (objective, latent_norm) pairs (e.g. to mix objectives
    without generating every combination): this replaces the
    --objective x --latent_norm product entirely.

Concurrency:
  - Up to --max_concurrent children run at once (subprocess.Popen + polling,
    no exotic deps). As children finish, new ones are started from the
    pending queue.
  - Before starting each new child, if CUDA_VISIBLE_DEVICES is set and
    --gpu_check is enabled (default), free memory on exactly the pinned
    GPU indices is queried via `nvidia-smi --query-gpu=... -i <indices>` and
    the launch is deferred (polled again later) until the minimum free MiB
    across those pinned indices is >= --min_free_mib. If
    CUDA_VISIBLE_DEVICES is unset (e.g. CPU-only dry runs), the check is
    skipped automatically.

Each child is invoked as:
    <this interpreter> -m latentdyn.train --latent_dim D --seed S --objective O
        --latent_norm NM --epochs E --log_every L --dense_early N
        --artifacts_root ROOT [--tag TAG] [--device DEV] [--smoke]
        [--max_steps K]
  (run_id is NOT passed explicitly -- the child derives the identical run_id
  itself via the same Config.resolve() formula, since it receives the same
  objective/latent_norm/latent_dim/seed/tag.)

Each child's combined stdout+stderr is captured to
<artifacts_root>/runs/<run_id>/train.log. A manifest listing every run_id,
its resolved launch config, and live status is written (and updated after
every state change) to:
    <artifacts_root>/sweeps/<sweep_name>/manifest.json

Usage:
    # coordinator has already done: export CUDA_VISIBLE_DEVICES=3
    python -m latentdyn.sweep --sweep_name capacity_sweep \\
        --latent_dims 2 4 8 16 32 --seeds 0 1 2 \\
        --objective probe --latent_norm none --max_concurrent 4

    # mixing objectives explicitly via --variant
    python -m latentdyn.sweep --sweep_name obj_sweep \\
        --variant probe:none --variant autoencoder:none --variant probe_recon:l2 \\
        --latent_dims 8 16 --seeds 0 1 --max_concurrent 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from latentdyn.config import Config, DEFAULT_ARTIFACTS_ROOT, VALID_LATENT_NORMS, VALID_OBJECTIVES, get_paths
from latentdyn.gpu import min_free_mib_on_pinned_gpus, pinned_gpu_indices
from latentdyn.launch import train_cmd


# --------------------------------------------------------------------------
# Grid / run_id
# --------------------------------------------------------------------------

def parse_variant(spec: str) -> Tuple[str, str]:
    """Parse one `--variant OBJECTIVE:LATENT_NORM` token."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--variant must look like OBJECTIVE:LATENT_NORM (got {spec!r})"
        )
    objective, latent_norm = spec.split(":", 1)
    if objective not in VALID_OBJECTIVES:
        raise argparse.ArgumentTypeError(
            f"--variant: unknown objective {objective!r} (expected one of {VALID_OBJECTIVES})"
        )
    if latent_norm not in VALID_LATENT_NORMS:
        raise argparse.ArgumentTypeError(
            f"--variant: unknown latent_norm {latent_norm!r} (expected one of {VALID_LATENT_NORMS})"
        )
    return objective, latent_norm


def derive_run_id(objective: str, latent_norm: str, latent_dim: int, seed: int, tag: Optional[str]) -> str:
    """Single source of truth for run_id: delegates to Config.resolve() so this
    can never drift from what the child `latentdyn.train` process will derive for
    itself. `device` is set to a harmless placeholder purely so `resolve()`
    skips its `import torch` auto-detect branch -- this process never touches
    torch/CUDA.
    """
    cfg = Config(
        objective=objective, latent_norm=latent_norm, latent_dim=latent_dim,
        seed=seed, tag=tag, device="cpu",
    )
    cfg.resolve()
    assert cfg.run_id is not None
    return cfg.run_id


def build_grid(
    variants: List[Tuple[str, str]], latent_dims: List[int], seeds: List[int]
) -> List[Dict]:
    """Cartesian product of variants x latent_dims x seeds -> list of job dicts."""
    jobs = []
    for (objective, latent_norm), latent_dim, seed in itertools.product(variants, latent_dims, seeds):
        jobs.append({"objective": objective, "latent_norm": latent_norm, "latent_dim": latent_dim, "seed": seed})
    return jobs


# --------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------

def write_manifest(manifest_path: Path, manifest: Dict) -> None:
    """Write the manifest atomically (write to a tmp file, then rename)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_path, manifest_path)


# --------------------------------------------------------------------------
# Child launch
# --------------------------------------------------------------------------

def build_child_cmd(job: Dict, args: argparse.Namespace) -> List[str]:
    cmd = train_cmd(
        "--latent_dim", str(job["latent_dim"]),
        "--seed", str(job["seed"]),
        "--objective", job["objective"],
        "--latent_norm", job["latent_norm"],
        "--epochs", str(args.epochs),
        "--log_every", str(args.log_every),
        "--dense_early", str(args.dense_early),
        "--artifacts_root", args.artifacts_root,
    )
    if args.tag:
        cmd += ["--tag", args.tag]
    if args.device:
        cmd += ["--device", args.device]
    if args.smoke:
        cmd += ["--smoke"]
    if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
    return cmd


def main() -> None:
    args = build_argparser().parse_args()

    # IMPORTANT: train.py's own config_from_args() defaults tag="smoke" when
    # --smoke is passed and --tag is unset (applied BEFORE run_id derivation).
    # Resolve that same effective tag HERE, once, and pass it explicitly to
    # every child (and use it for our own run_id derivation below) so the
    # run_id this process predicts can never drift from the run_id the child
    # actually resolves for itself.
    if args.tag is None and args.smoke:
        args.tag = "smoke"

    if args.variant:
        variants = [parse_variant(v) for v in args.variant]
    else:
        variants = list(itertools.product(args.objective, args.latent_norm))

    jobs = build_grid(variants, args.latent_dims, args.seeds)
    if not jobs:
        print("[run_sweep] empty grid -- nothing to do.")
        return

    # Attach run_id + run_dir + log_path to every job up front.
    for job in jobs:
        run_id = derive_run_id(job["objective"], job["latent_norm"], job["latent_dim"], job["seed"], args.tag)
        cfg_stub = Config(run_id=run_id, artifacts_root=args.artifacts_root)
        paths = get_paths(cfg_stub)
        job["run_id"] = run_id
        job["run_dir"] = str(paths.run_dir)
        job["log_path"] = str(paths.run_dir / "train.log")

    run_ids = [j["run_id"] for j in jobs]
    if len(set(run_ids)) != len(run_ids):
        dupes = sorted({r for r in run_ids if run_ids.count(r) > 1})
        raise SystemExit(
            f"[run_sweep] ERROR: grid produces duplicate run_ids (collision): {dupes}. "
            f"Pass a --tag to disambiguate, or check for duplicate (objective, latent_norm, latent_dim, seed) entries."
        )

    manifest_path = Path(args.artifacts_root) / "sweeps" / args.sweep_name / "manifest.json"
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(
        f"[run_sweep] sweep_name={args.sweep_name!r} n_jobs={len(jobs)} "
        f"max_concurrent={args.max_concurrent} CUDA_VISIBLE_DEVICES={cuda_visible!r} "
        f"(inherited from the coordinator's environment -- latentdyn.sweep never sets this itself) "
        f"manifest={manifest_path}"
    )

    manifest = {
        "sweep_name": args.sweep_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_visible_devices_at_launch": cuda_visible,
        "max_concurrent": args.max_concurrent,
        "min_free_mib": args.min_free_mib,
        "gpu_check": args.gpu_check,
        "runs": {},
    }
    for job in jobs:
        manifest["runs"][job["run_id"]] = {
            "run_id": job["run_id"],
            "config": {
                "objective": job["objective"],
                "latent_norm": job["latent_norm"],
                "latent_dim": job["latent_dim"],
                "seed": job["seed"],
                "epochs": args.epochs,
                "log_every": args.log_every,
                "dense_early": args.dense_early,
                "device": args.device,
                "tag": args.tag,
                "smoke": args.smoke,
                "max_steps": args.max_steps,
                "artifacts_root": args.artifacts_root,
            },
            "run_dir": job["run_dir"],
            "log_path": job["log_path"],
            "status": "pending",
            "pid": None,
            "returncode": None,
            "started_at": None,
            "finished_at": None,
        }
    write_manifest(manifest_path, manifest)

    pending = list(jobs)
    running: List[Dict] = []  # each: {"job": job, "proc": Popen, "log_fh": file}

    def launch(job: Dict) -> None:
        run_dir = Path(job["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(job["log_path"], "w")
        cmd = build_child_cmd(job, args)
        print(f"[run_sweep] launching {job['run_id']}: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),  # inherit the FULL environment unchanged (incl. CUDA_VISIBLE_DEVICES)
            )
        except Exception as e:  # noqa: BLE001
            log_fh.close()
            entry = manifest["runs"][job["run_id"]]
            entry["status"] = "failed_to_launch"
            entry["returncode"] = None
            entry["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            write_manifest(manifest_path, manifest)
            print(f"[run_sweep] ERROR launching {job['run_id']}: {e}")
            return
        entry = manifest["runs"][job["run_id"]]
        entry["status"] = "running"
        entry["pid"] = proc.pid
        entry["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_manifest(manifest_path, manifest)
        running.append({"job": job, "proc": proc, "log_fh": log_fh})

    try:
        while pending or running:
            # Reap finished children.
            still_running = []
            for r in running:
                ret = r["proc"].poll()
                if ret is None:
                    still_running.append(r)
                    continue
                r["log_fh"].close()
                entry = manifest["runs"][r["job"]["run_id"]]
                entry["status"] = "done" if ret == 0 else "failed"
                entry["returncode"] = ret
                entry["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                write_manifest(manifest_path, manifest)
                print(f"[run_sweep] finished {r['job']['run_id']} returncode={ret}")
            running[:] = still_running

            # Start new children while slots are free and (optionally) the
            # pinned GPU(s) have enough free memory.
            while pending and len(running) < args.max_concurrent:
                if args.gpu_check:
                    free_mib = min_free_mib_on_pinned_gpus()
                    if free_mib is not None and free_mib < args.min_free_mib:
                        print(
                            f"[run_sweep] waiting: free GPU mem {free_mib} MiB < "
                            f"--min_free_mib {args.min_free_mib} MiB on pinned device(s) "
                            f"{pinned_gpu_indices()}"
                        )
                        break
                job = pending.pop(0)
                launch(job)

            if pending or running:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("[run_sweep] KeyboardInterrupt -- terminating running children and releasing GPU memory...")
        for r in running:
            r["proc"].terminate()
        for r in running:
            try:
                r["proc"].wait(timeout=15)
            except subprocess.TimeoutExpired:
                r["proc"].kill()
                r["proc"].wait(timeout=15)
            r["log_fh"].close()
            entry = manifest["runs"][r["job"]["run_id"]]
            entry["status"] = "interrupted"
            entry["returncode"] = r["proc"].returncode
            entry["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        for job in pending:
            manifest["runs"][job["run_id"]]["status"] = "cancelled"
        write_manifest(manifest_path, manifest)
        raise

    n_done = sum(1 for v in manifest["runs"].values() if v["status"] == "done")
    n_failed = sum(1 for v in manifest["runs"].values() if v["status"] not in ("done",))
    print(f"[run_sweep] sweep {args.sweep_name!r} complete: {n_done} done, {n_failed} not-done. manifest={manifest_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Launch a grid of latentdyn.train runs as co-located subprocesses on whatever GPU(s) "
            "are already pinned via CUDA_VISIBLE_DEVICES (set by the coordinator -- this "
            "script NEVER sets that variable itself)."
        )
    )
    p.add_argument("--sweep_name", type=str, required=True, help="Used to namespace the manifest under <artifacts_root>/sweeps/<sweep_name>/.")
    p.add_argument("--latent_dims", type=int, nargs="+", default=[16], help="Grid axis: model capacity.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="Grid axis: random seed.")
    p.add_argument(
        "--objective", type=str, nargs="+", default=["probe"], choices=list(VALID_OBJECTIVES),
        help="Grid axis (crossed with --latent_norm unless --variant is given).",
    )
    p.add_argument(
        "--latent_norm", type=str, nargs="+", default=["none"], choices=list(VALID_LATENT_NORMS),
        help="Grid axis (crossed with --objective unless --variant is given).",
    )
    p.add_argument(
        "--variant", type=str, action="append", default=None, metavar="OBJECTIVE:LATENT_NORM",
        help="Repeatable explicit (objective, latent_norm) pair, e.g. "
             "--variant probe:none --variant autoencoder:l2. If given at least once, "
             "REPLACES the --objective x --latent_norm cartesian product with exactly "
             "these pairs (so one invocation can mix objectives without the full cross).",
    )
    p.add_argument("--dense_early", type=int, default=0, help="Forwarded to every child's --dense_early.")
    p.add_argument("--log_every", type=int, default=50, help="Forwarded to every child's --log_every.")
    p.add_argument("--epochs", type=int, default=5, help="Forwarded to every child's --epochs.")
    p.add_argument("--max_concurrent", type=int, default=4, help="Max number of train.py children running at once.")
    p.add_argument(
        "--min_free_mib", type=int, default=2000,
        help="Before launching a new child, wait until free memory on the pinned GPU(s) "
             "(per CUDA_VISIBLE_DEVICES) is at least this many MiB. Ignored if "
             "CUDA_VISIBLE_DEVICES is unset or --no_gpu_check is passed.",
    )
    gpu_check = p.add_mutually_exclusive_group()
    gpu_check.add_argument("--gpu_check", dest="gpu_check", action="store_true", default=True, help="Enable the --min_free_mib guard (default).")
    gpu_check.add_argument("--no_gpu_check", dest="gpu_check", action="store_false", help="Disable the GPU memory guard entirely.")
    p.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Forwarded to every child's --device (NOT a GPU selector -- that is "
             "CUDA_VISIBLE_DEVICES, which this script never touches). Default: omitted, "
             "so each child auto-detects.",
    )
    p.add_argument("--smoke", action="store_true", help="Forward --smoke to every child (tiny/fast configs for a dry run).")
    p.add_argument("--max_steps", type=int, default=None, help="Forwarded to every child's --max_steps.")
    p.add_argument("--tag", type=str, default=None, help="Optional common tag suffix forwarded to every child's run_id (disambiguates repeated sweeps).")
    p.add_argument("--artifacts_root", type=str, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--poll_interval", type=float, default=2.0, help="Seconds between polling loops.")
    return p


if __name__ == "__main__":
    main()
