# v2 slingshot C, SGD D, epsilon F1/F2 and SGD-lr G

**Date**: 2026-10-15 11:54:59 UTC
**Status**: back-filled 2026-10-26 from existing artifacts
**Datasets**: mnist
**Tags**: back-fill, v2-optimizer-sweeps

Date source: earliest verified config/report artifact mtime in the inspected scope: `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs/mnist_probe_none_d16_s1_sl40k_n250_wd0/config.json` = `2026-10-15T11:54:59.816930+00:00`. This is filesystem evidence, not a recovered launch timestamp. See [config provenance](data/config-provenance.csv) and [artifact provenance](data/artifact-provenance.csv).

## Goal

Test optimizer, epsilon and learning-rate dependence of loss spikes and class separation.

## Method

40k-step Adam C and momentum-SGD D; epsilon F1/F2 span 1e-8 to 1e-3; G varies SGD lr .0125/.025/.05/.1. Use jobs_C/D/F1/F2/G.txt and run_queue.sh under the absolute runs_v2 artifact root. Seeds and per-run overrides are in the registry.

Historical execution was on coeus. Exact historical GPU/PID and launch environment are unknown unless a preserved source says otherwise; no launches were made for this back-fill. Registry commands are reconstructed reproductions, not launch logs.

## Runs

**Count**: 372 registry rows.

[Registry](../results/runs.csv), [sweep index](../results/sweeps.csv): filter `group in ['runs_v2/C', 'runs_v2/D', 'runs_v2/F1', 'runs_v2/F2', 'runs_v2/G']`. Counts describe registered runs, not a fresh completion audit.

| Registry group | Count | Registry status |
|---|---:|---|
| `runs_v2/C` | 90 | complete: 90 |
| `runs_v2/D` | 30 | complete: 30 |
| `runs_v2/F1` | 90 | complete: 90 |
| `runs_v2/F2` | 90 | complete: 90 |
| `runs_v2/G` | 72 | complete: 72 |

Resolved configs are `/vol/bitbucket/lbastian/latent-dynamics/<rel_path>/config.json`, using each selected registry `rel_path`. Commands, datasets, seeds and overrides are in the same rows; `results/config_defaults.json` records omitted defaults. All 372 selected config paths were stat-checked; their combined file size is 376686 bytes. Example config: `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs/mnist_probe_none_d16_s0_sl40k_n1000_wd0/config.json`.



Absolute coeus artifacts (retained remotely):

- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs`
- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/jobs_C.txt`
- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/jobs_D.txt`
- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/jobs_F1.txt`
- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/jobs_F2.txt`
- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/jobs_G.txt`

No recursive size walk was used. Where a bulk size is unavailable, it remains unmeasured; source-file sizes below and provenance CSVs are exact metadata observations.

## Results

dynamics_report/report.tex reports stability score AUC 0.59 and no SGD spikes. Later dense/analysis/REPORT.md finds 4–8 spikes per wd1e-4 run in 40k steps, peaks .6–3, without epsilon-floor involvement; wd0 type N has 1–2 spikes with peaks 8–19. Snapshot event fractions from the original grids do not measure whether a run spikes.

Unless explicitly called a direct metadata/table check, headline numbers below are copied from the cited reports and **not re-verified against raw data**. Original reports can contain superseded recommendations; this entry records the later controls and caveats.

Sources (links are local preserved copies; absolute paths identify the originals on coeus):

| Source | Absolute coeus path | Bytes | mtime UTC |
|---|---|---:|---|
| [dynamics_report/report.tex](data/dynamics_report/report.tex) | `/vol/bitbucket/lbastian/latent-dynamics/dynamics_report/report.tex` | 41432 | 2026-10-26T18:35:37.302217+00:00 |
| [dynamics_v2/dense/analysis/REPORT.md](data/dynamics_v2/dense/analysis/REPORT.md) | `/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense/analysis/REPORT.md` | 2716 | 2026-10-16T11:48:35.643461+00:00 |

## Conclusion

SGD and Adam differ; the epsilon-floor account covers type N, not repeated type F spikes.

## Next Steps

Any repeat must log per-step minibatch loss and gradient norms; do not rerun the entire grid on snapshot-only evidence.

## Code revision

unknown; current code at `ec5b452481b89b6901a07f4cb660c66eb6d06e29`. Registry published by `f426cb6c9179e76eaa72202ea750271da1d28a1f`; this does not establish which revision produced historical runs. Source reports may name pre-refactor paths; current entry points are under `scripts/`, reusable modules under `src/latentdyn/`, frozen analyses under `archive/`.
