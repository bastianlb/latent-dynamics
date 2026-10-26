# CLAUDE.md

## Agent orchestration (model usage)

**Opus orchestrates only — it does not implement.** All substantive
implementation work (writing/editing code, running experiments, analysis,
debugging) is delegated to **sonnet or haiku subagents**; Opus plans,
coordinates, launches GPU/analysis jobs, reviews results, and talks to the user.
Default to the cheapest model that fits the task (haiku for small/mechanical
jobs, sonnet for non-trivial implementation). Spawn agents with an explicit
`model: sonnet` / `model: haiku` override. Only fall back to doing work inline
when delegation is impractical (e.g. a quick coordinator-level shell check).

## Code layout

- New reusable code goes into `src/latentdyn/` (installed editable: `pip install -e .`).
- Entry points / thin wrappers go into `scripts/` (`scripts/ae/` for the AE study).
- `archive/` is frozen one-off analysis code: do not extend it.
- Equivalence/self tests live in `tests/`; run everything with `bash tests/run_all.sh`.

## GPU usage policy (shared machine)

This is a **shared machine** with 8× NVIDIA L40S GPUs (46 GB each). Other people
run jobs here. Follow these rules strictly for anything that touches a GPU.

### Rules

1. **Only use idle GPUs.** Never launch work on a GPU that someone else is using.
2. **Max 3 GPUs at a time** (raised from 2 by the user, 2026-10-16). Never occupy more than three GPUs across all running work.
3. **Always pin explicitly.** Set `CUDA_VISIBLE_DEVICES` to the specific idle GPU
   index/indices you selected. Never let a job grab whatever it wants or spill
   across all GPUs.
4. **Check before every launch.** GPU occupancy changes constantly on a shared
   box — re-check immediately before starting any GPU job, not once per session.

### What counts as "idle"

Check with:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

A GPU is **idle** and safe to use only if **both**:

- `memory.used` is near zero (treat < ~500 MiB as free; anything higher means
  someone has a process resident even if utilization reads 0%), and
- `utilization.gpu` is ~0%.

If no GPU meets this, **do not launch** — tell the user the machine is busy and
wait, rather than squeezing onto an occupied GPU.

### How to run

Pick idle indices and pin them. Example for a single idle GPU (index 3):

```bash
CUDA_VISIBLE_DEVICES=3 python -m latentdyn.train
```

Two idle GPUs (indices 0 and 3) — the maximum allowed:

```bash
CUDA_VISIBLE_DEVICES=0,3 python -m latentdyn.train
```

Do not use more than three indices in `CUDA_VISIBLE_DEVICES`.

### Stacking several of our own small jobs on one GPU

Our models here are tiny (MNIST-scale, well under ~1 GB VRAM each). It is fine to
run **multiple of our own runs on a single GPU concurrently** — e.g. several
`latent_dim`/seed variants pinned to the same idle index — **as long as the
combined `memory.used` stays comfortably within the 46 GB budget** (leave
headroom; don't fill the card). This does not relax the other rules: still only
land on GPUs that are idle for *others*, still never span more than three GPU
indices total, and re-check occupancy before each launch.

### Release promptly

Being a good citizen on a shared box means freeing GPUs as soon as the work is
done — don't sit on memory others could use.

- **Don't leave idle processes resident.** When a job finishes (or is killed),
  make sure its process actually exits and releases GPU memory. A crashed or
  backgrounded Python process can keep a GPU's memory reserved while reading 0%
  utilization — that blocks others.
- **Verify the GPU is free after stopping** a job:

  ```bash
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  ```

  If memory is still held on an index you used, find and kill the stray process:

  ```bash
  nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv
  kill <pid>
  ```

- **Don't hold GPUs while idle.** Avoid keeping notebooks, REPLs, or debug
  sessions attached to a GPU when you're not actively computing — release and
  re-acquire instead.

## CPU usage (shared machine — 128 cores)

This box has **128 CPU cores shared with others**. numpy/scipy/BLAS and ripser
default to spawning a thread per core, and several processes doing that at once
**oversubscribe** the machine (load average >> 128, everything crawls). This has
already bitten us: an uncapped analysis run burned ~465 CPU-minutes and drove
load to ~180.

**Always cap thread pools for CPU-heavy numpy/BLAS/TDA work.** Set these
*before* numpy is imported (env vars, or `setdefault` at the top of the entry
script so they stay overridable):

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 python -m latentdyn.pipeline.analyze ...
```

`latentdyn.pipeline.analyze` / `latentdyn.pipeline.plot` already apply this cap internally (via `latentdyn._threads.cap_threads`). For any new
CPU-bound script, do the same (≤8 threads is plenty here), and sanity-check
`uptime` / `/proc/loadavg` after launching.

## Storage policy (shared drives)

Drives fill up and are shared. Check free space before writing anything bulky:

```bash
df -h /data /data2 /vol/bitbucket
```

Known layout and rules:

- **`/data` — nearly full (~28 G free, 95%+). Do NOT write here.** Not the env,
  not data, not outputs.
- **`/data2` — local, faster, ~400 G free but 97% used.** Use for the **conda
  environment** (consistent with existing envs under
  `/data2/lbastian/conda/envs/`). Exception (user, 2026-10-16): the dense
  optimizer-state dumps (`--log_opt_dense` / `--dump_opt_every` / `--dump_window`)
  go to the local server folder `/data2/lbastian/latent-dynamics/opt_dense/`, not
  bitbucket. Cap them at ~40 G total, check `df -h /data2` before each batch, and
  delete superseded dumps. Other bulky run artifacts stay OFF this drive.
- **`/vol/bitbucket` — network (NFS), huge (~26 T free). Use for all bulky
  artifacts:** dataset caches, latent dumps, checkpoints, figures. Project
  artifacts live under `/vol/bitbucket/lbastian/latent-dynamics/`.
- **`/homes/lbastian/latent-dynamics` — code only** (small, version-controlled
  working tree). Point code at the bitbucket paths via config/env var; never
  dump large files into the home working tree.

### Rules

1. **Never write large/growing outputs to `/data` or `/homes`.** Send them to
   `/vol/bitbucket/lbastian/latent-dynamics/`.
2. **Estimate before you dump.** Before any run that writes many files or large
   arrays, estimate total size (per-checkpoint size × count) and state it. This
   project can accumulate a lot of per-iteration latent snapshots — only log a
   fixed probe subset, prefer compact dtypes (float16/float32), and avoid saving
   full-dataset latents or dense model checkpoints every step.
3. **Clean up intermediate junk.** Remove superseded dumps/checkpoints rather
   than letting them pile up.

## Conda environment

- Project env: **`latent-dyn`** at `/data2/lbastian/conda/envs/latent-dyn`
  (Python 3.11). Create/solve with `mamba` (`/data/lbastian/miniconda3/bin/mamba`).
- Activate with the full prefix:
  `conda activate /data2/lbastian/conda/envs/latent-dyn`.
- Core stack: PyTorch (+CUDA) + torchvision, numpy/scipy/scikit-learn/matplotlib,
  and TDA libs (`ripser`, `persim`, `gudhi`) for persistent-homology metrics.


## Laboratory notebook

Before experiment work, read [notebook/README.md](notebook/README.md) and
[notebook/WORKING.md](notebook/WORKING.md). Follow the agentmemory laboratory
notebook process, including the existing-project back-fill rules (2026-10-26):
search prior entries/results/run indexes before repeating work; preserve negative
results and append dated corrections rather than silently revising numbers.

Every direct coeus launch gets a dated entry and WORKING.md update before or at
launch: exact command, configs/overrides, dataset/splits/seeds, code revision or
runtime hashes, owner, GPU index/PID, interpreter, log/output paths and expected
completion evidence. Record the PID immediately; distinguish pending/startup from
running, and update both records on progress, completion, failure or blockage. Pin
code for long runs; do not edit an active run's code. Verify timestamped logs and
expected artifacts, not just a PID, mtime or exit status. Delegates receive the
entry path and these rules. Keep bulk artifacts on the approved coeus storage;
small reports/registries belong in git and point back to absolute source paths.
