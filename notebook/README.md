# Laboratory notebook

This git-tracked notebook is the system of record for latent-dynamics experiments.
Read this file and [WORKING.md](WORKING.md) before experiment work; search existing
entries, results and cross-worktree run indexes before repeating a run. Justify
additional seeds or duplicates. Dated entries preserve history; WORKING is the concise
live index of running/pending work, recent completions, blockers and next steps.

## Naming and provenance

Use `YYYY-MM-DD_HHMMSS_short-name.md`, normally the first launch time. The initial
back-fill uses the earliest verified artifact/run mtime in UTC within the stated
inventory scope, not an inferred launch time. Entries explicitly name that source;
mtimes may reflect copying and do not prove launch or completion. Status is
`back-filled 2026-10-26 from existing artifacts`. Historical launch commands and
producing revisions remain unknown unless recoverable; reconstructed registry
commands are not contemporaneous launch logs.

The existing [run registry](../results/runs.csv) and [sweep index](../results/sweeps.csv)
are kept in results, without duplicates. Registry status is recorded evidence, not a
fresh audit of every checkpoint. Small source reports are preserved in data with
absolute coeus paths, byte counts and checksums in the copy manifest. Raw latent,
optimizer and checkpoint files remain on coeus. Distinguish copied claims (not
re-verified against raw data), new direct checks, estimates and single-seed results.
Never silently change old numbers: append dated corrections and preserve null,
negative, failed and blocked results. Analysis families reuse runs; entry counts
must not be summed as unique training runs.

## Launch and completion rules

Before or at every launch, create/update both the dated entry and WORKING: exact
command, configs and overrides, revision/runtime hashes, dataset/splits/seeds,
resource owner, GPU index/PID, interpreter, log/output paths and expected completion
evidence. Record the PID immediately; distinguish pending, startup, running,
complete and failed. Pin code for long runs and do not edit code used by active jobs.
Update both records as evidence changes. A PID, mtime or exit code alone does not
prove progress or success: inspect timestamped logs and expected final artifacts.
When delegating, give workers the entry path and notebook rules. Papers, memory
and dashboards link here for evidence.

Compute uses direct launches on coeus, without SLURM. Follow [CLAUDE.md](../CLAUDE.md):
only GPUs idle for others (<~500 MiB and ~0% utilization), check before each launch,
explicit CUDA_VISIBLE_DEVICES, at most three GPU indices total; several small owned
runs may share one otherwise idle GPU with memory headroom. CPU-heavy analysis caps
OMP/OPENBLAS/MKL/NUMEXPR/VECLIB thread pools at eight or fewer before imports.
Interpreter: `/data2/lbastian/conda/envs/latent-dyn/bin/python`. Code is in
`/homes/lbastian/latent-dynamics`; bulky artifacts go to
`/vol/bitbucket/lbastian/latent-dynamics`. Never put growing outputs in home or
`/data`. The approved optimizer-dump exception is
`/data2/lbastian/latent-dynamics/opt_dense` (about 40 GB total; disk guards and
pre-launch free-space checks apply). Estimate storage before dumping; limit probe
size and precision. Free resources promptly, respecting each task's permissions.
This back-fill authorized no compute, GPU work, process termination or artifact writes.

## Entry template

```markdown
# Experiment title

**Date**: YYYY-MM-DD HH:MM:SS UTC (first launch; back-fills name date evidence)
**Status**: Running / Complete / Failed / Blocked
**Datasets**: ...
**Tags**: ...

## Goal
## Method
<!-- Design, split/probe, seeds, environment; measured versus inferred. -->
## Runs
<!-- Count, registry filter, exact command/config overrides, GPU/PID/owner,
     logs, absolute coeus outputs, storage estimate and completion evidence. -->
## Results
<!-- Numbers with source paths; nulls/failures; verification limits. -->
## Conclusion
## Next Steps
## Code revision
<!-- Producing SHA/runtime hash, or unknown; current code at SHA. -->
```

The anisotropy template requested for this adoption was absent at
`/Users/lbastian/coding/anisotropy/notebook/README.md`; the read-only cortical-shape
matching template and agentmemory notebook process (2026-10-26) informed these rules.
