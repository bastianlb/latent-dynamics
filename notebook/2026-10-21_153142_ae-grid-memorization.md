# AE grid, memorization and controls

**Date**: 2026-10-21 15:31:42 UTC
**Status**: back-filled 2026-10-26 from existing artifacts
**Datasets**: cifar10, fashion_mnist, mnist
**Tags**: back-fill, ae-grid-memorization

Date source: earliest verified config/report artifact mtime in the inspected scope: `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/runs/mnist_ae_mlp_d16_s0_nfull_wd1e-4/config.json` = `2026-10-21T15:31:42.964682+00:00`. This is filesystem evidence, not a recovered launch timestamp. See [config provenance](data/config-provenance.csv) and [artifact provenance](data/artifact-provenance.csv).

## Goal

Relate AE overfitting and memorization to latent geometry, with sample-size, PCA and linear-AE controls.

## Method

MNIST/FM MLP AE and linear AE, N250/1000/4000/full, wd0/1e-4, dimensions8/16/64, seeds0–2; CIFAR conv/linear controls (42 runs). Grid run_ae_grid.py, analysis latentdyn.ae and scripts/ae; two smoke runs tracked separately. Fixed train/test probes; labels are annotations.

Historical execution was on coeus. Exact historical GPU/PID and launch environment are unknown unless a preserved source says otherwise; no launches were made for this back-fill. Registry commands are reconstructed reproductions, not launch logs.

## Runs

**Count**: 138 complete grid runs + 2 complete smoke runs.

[Registry](../results/runs.csv), [sweep index](../results/sweeps.csv): filter `group in ['ae_dynamics', 'ae_dynamics/smoke']`. Counts describe registered runs, not a fresh completion audit.

| Registry group | Count | Registry status |
|---|---:|---|
| `ae_dynamics` | 138 | complete: 138 |
| `ae_dynamics/smoke` | 2 | complete: 2 |

Resolved configs are `/vol/bitbucket/lbastian/latent-dynamics/<rel_path>/config.json`, using each selected registry `rel_path`. Commands, datasets, seeds and overrides are in the same rows; `results/config_defaults.json` records omitted defaults. All 140 selected config paths were stat-checked; their combined file size is 205331 bytes. Example config: `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/runs/cifar10_ae_conv_d16_s0_n1000_wd0/config.json`.

Registry dataset counts for the main grid: MNIST48, FM48, CIFAR42; do not add dimension ablations to 138 again. FINDINGS_cifar10.md was not found; use report.tex and analysis/cifar10 outputs.

Absolute coeus artifacts (retained remotely):

- `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/runs`
- `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis`
- `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/cifar10`

No recursive size walk was used. Where a bulk size is unavailable, it remains unmeasured; source-file sizes below and provenance CSVs are exact metadata observations.

## Results

FINDINGS.md/FINDINGS_fashion.md and report.tex: gap opens30–50 epochs; MNIST late rise+63/+52/+4.3% at N250/1k/4k, FM+48/+34/+3.5%, CIFAR+76/+93/+61%. At MNIST N1k, error-typicality rho+.03 train versus+.70 test; report controls show |train-test difference|<=.03 for linear/PCA versus-.6 to-.78 for MLP at small N on MNIST/FM. Pixel-PCA reproduces spread1.17 vs observed1.20 at N250. NN_RATIO_CHECK.md retracts N-leveraged correlations: N-dummy partial-.29 FM,-.07 MNIST; CIFAR-.56 remains unresolved.

Unless explicitly called a direct metadata/table check, headline numbers below are copied from the cited reports and **not re-verified against raw data**. Original reports can contain superseded recommendations; this entry records the later controls and caveats.

Sources (links are local preserved copies; absolute paths identify the originals on coeus):

| Source | Absolute coeus path | Bytes | mtime UTC |
|---|---|---:|---|
| [ae_dynamics/analysis/FINDINGS.md](data/ae_dynamics/analysis/FINDINGS.md) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/FINDINGS.md` | 14125 | 2026-10-21T17:55:28.422024+00:00 |
| [ae_dynamics/analysis/FINDINGS_fashion.md](data/ae_dynamics/analysis/FINDINGS_fashion.md) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/FINDINGS_fashion.md` | 12137 | 2026-10-21T20:14:07.170129+00:00 |
| [ae_dynamics/analysis/NN_RATIO_CHECK.md](data/ae_dynamics/analysis/NN_RATIO_CHECK.md) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/NN_RATIO_CHECK.md` | 3860 | 2026-10-21T20:13:48.549842+00:00 |
| [ae_dynamics/analysis/summary_by_config.md](data/ae_dynamics/analysis/summary_by_config.md) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/summary_by_config.md` | 8835 | 2026-10-21T17:20:08.352280+00:00 |
| [dynamics_report/report.tex](data/dynamics_report/report.tex) | `/vol/bitbucket/lbastian/latent-dynamics/dynamics_report/report.tex` | 41432 | 2026-10-26T18:35:37.302217+00:00 |

## Conclusion

Decoder-tied error decoupling survives relevant controls on MNIST/FM; global spread and N-driven correlations are not memorization mechanisms.

## Next Steps

Calibrate difficulty controls; resolve CIFAR partial correlation and matched-classifier comparisons before strong cross-architecture conclusions.

## Code revision

unknown; current code at `ec5b452481b89b6901a07f4cb660c66eb6d06e29`. Registry published by `f426cb6c9179e76eaa72202ea750271da1d28a1f`; this does not establish which revision produced historical runs. Source reports may name pre-refactor paths; current entry points are under `scripts/`, reusable modules under `src/latentdyn/`, frozen analyses under `archive/`.
