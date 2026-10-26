# Archive

Frozen historical analyses from 2026-10-14 to 2026-10-22, kept for provenance.

- These scripts are not maintained and may not run against the current code.
- Where they use shared code, they import the `latentdyn` package (`pip install -e .`).
- Intra-archive imports (`common`, `common2`, `stats_util`, `step*_lib`, etc.) still refer to sibling files in the same directory.

Subdirectories:

- `analysis_v2/` - v2 statistical analysis pipeline (steps 2-10, ANALYSIS_PLAN.md).
- `reanalysis/` - reanalysis scripts and issue checks, with run_all.sh.
- `paper_v2_figs/` - figure and table generators for the v2 paper draft.
- `slingshot/` - phase-diagram, stability, per-class, SGD learning-rate and NC1 checks.
- `dense/` - dense-optimizer-state lead/lag analyses.
- `misc/` - inventory, summary figures and linear-AE theory plots.
- `aggregates/` - aggregate tables over vfield and NTK runs.
