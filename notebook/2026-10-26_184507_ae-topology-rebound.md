# AE topology rebound and regime geometry

**Date**: 2026-10-26 18:45:07 UTC
**Status**: back-filled 2026-10-26 from existing artifacts
**Datasets**: fashion_mnist, mnist
**Tags**: back-fill, ae-topology-rebound

Date source: earliest verified family output artifact mtime in the inspected scope: `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/topo/fashion_mnist_ae_mlp_d16_s2_n1000_wd0.npz` = `2026-10-26T18:45:07.722276+00:00`. This is filesystem evidence, not a recovered launch timestamp. See [config provenance](data/config-provenance.csv) and [artifact provenance](data/artifact-provenance.csv).

## Goal

Determine whether topological rebound tracks overfitting or general fitting progress.

## Method

MNIST/FM,AE MLPd16 wd0/1e-4 and linear d16wd0, N250/1k/4k/full,3seeds; classifiers phasewd0 include N16k. H1 entropy/lifetime/PMag on66 masked snapshots; last snapshot excluded; minimum searched step>=2,end median step>4000. scripts/ae/ae_rebound_corr.py and ae_regime_metrics_fig.py.

Historical execution was on coeus. Exact historical GPU/PID and launch environment are unknown unless a preserved source says otherwise; no launches were made for this back-fill. Registry commands are reconstructed reproductions, not launch logs.

## Runs

**Count**: 102 reused runs in rebound_numbers.json: 72 AE +30 classifier.

[Registry](../results/runs.csv), [sweep index](../results/sweeps.csv): filter `group in ['ae_dynamics', 'runs_v2/mnist_block_A', 'runs_v2/FM']` plus the restriction below. Counts describe registered runs, not a fresh completion audit.

| Registry group | Count | Registry status |
|---|---:|---|
| `ae_dynamics` | 72 | complete: 72 |
| `runs_v2/mnist_block_A` | 15 | complete: 15 |
| `runs_v2/FM` | 15 | complete: 15 |

Resolved configs are `/vol/bitbucket/lbastian/latent-dynamics/<rel_path>/config.json`, using each selected registry `rel_path`. Commands, datasets, seeds and overrides are in the same rows; `results/config_defaults.json` records omitted defaults. All 102 selected config paths were stat-checked; their combined file size is 135777 bytes. Example config: `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/runs/fashion_mnist_ae_linear_d16_s0_n1000_wd0/config.json`.

Registry restriction: AE MNIST/FM d16 MLP wd0/1e-4 or linear wd0,N250/1k/4k/full (72); classifier phase d16wd0,N250/1k/4k/16k/full (30). Match exact run IDs against rebound_numbers.json/runs. Train NPZs exist for 18 N=1k, 12 full-N, 6 N=4k and 6 N=250 runs; all 72 test NPZs exist. Geometry has 78 caches: 72 AE + 6 N=1k classifiers; other classifier geometry was not found in this cache. This expands the earlier24-AE-run,test-only regime summary.

Absolute coeus artifacts (retained remotely):

- `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/topo`
- `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/geom`
- `/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/cache`
- `/vol/bitbucket/lbastian/latent-dynamics/dynamics_report/figures`

No recursive size walk was used. Where a bulk size is unavailable, it remains unmeasured; source-file sizes below and provenance CSVs are exact metadata observations.

## Results

REBOUND_VERDICT.md: H1 total minimum step7–8,half recovery56–70 across N; gap onset85/266/1400 steps and never at fullN. At N1k minimum epoch.94,e50=8.7,gap onset33; relative rebound linear.58 vs MLP.61. Pooled48 MLP runs rebound-gap rho-.08,within-N .14; gap-logN rho-.97. wd at N1k cuts test excess to.42x and gap.74x while rebound1.1x. Train/test H1 relative difference .4% before/-1.2% after onset. Direct JSON/table check:102 runs=48MLP+24linear+30classifier; H1 e_min step-clock ratio1.0667,e50 1.2411, consistent with verdict1.07/1.24. Direct directory stat:72test+42train topology NPZs (not all train caches exist),4,117,539bytes across119direct files;78geometry caches,5,790,505bytes across81files. REGIME_METRICS_TAKEAWAYS.md: at N1k MNIST PR dimension10->1.2->6.4->8.4 across init/early-fit/fitting/end; class/neighborhood structure is nearly flat once gap opens. H1 e90 step ratio10.19, so late tail is not step-clock invariant.

Unless explicitly called a direct metadata/table check, headline numbers below are copied from the cited reports and **not re-verified against raw data**. Original reports can contain superseded recommendations; this entry records the later controls and caveats.

Sources (links are local preserved copies; absolute paths identify the originals on coeus):

| Source | Absolute coeus path | Bytes | mtime UTC |
|---|---|---:|---|
| [ae_dynamics/analysis/topo/REGIMES_TAKEAWAYS.md](data/ae_dynamics/analysis/topo/REGIMES_TAKEAWAYS.md) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/topo/REGIMES_TAKEAWAYS.md` | 5363 | 2026-10-26T19:24:58.658759+00:00 |
| [ae_dynamics/analysis/topo/REBOUND_VERDICT.md](data/ae_dynamics/analysis/topo/REBOUND_VERDICT.md) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/topo/REBOUND_VERDICT.md` | 3934 | 2026-10-26T22:50:30.344001+00:00 |
| [ae_dynamics/analysis/topo/REBOUND_TAKEAWAYS.md](data/ae_dynamics/analysis/topo/REBOUND_TAKEAWAYS.md) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/topo/REBOUND_TAKEAWAYS.md` | 41054 | 2026-10-26T22:54:04.725871+00:00 |
| [ae_dynamics/analysis/topo/rebound_numbers.json](data/ae_dynamics/analysis/topo/rebound_numbers.json) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/topo/rebound_numbers.json` | 340984 | 2026-10-26T22:54:04.911874+00:00 |
| [ae_dynamics/analysis/geom/REGIME_METRICS_TAKEAWAYS.md](data/ae_dynamics/analysis/geom/REGIME_METRICS_TAKEAWAYS.md) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/geom/REGIME_METRICS_TAKEAWAYS.md` | 7822 | 2026-10-26T22:49:36.460279+00:00 |
| [ae_dynamics/analysis/geom/regime_metric_table.csv](data/ae_dynamics/analysis/geom/regime_metric_table.csv) | `/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/geom/regime_metric_table.csv` | 27546 | 2026-10-26T22:45:56.820340+00:00 |

## Conclusion

Early rebound follows optimizer-time fitting and dimension recovery, not AE overfitting. Train/test topology broadly agrees; small late tails and classifier contrasts retain caveats.

## Next Steps

Resolve classifier final-snapshot outlier; control dataset/N and probe size in correlations. At N250 train probe has250points and is excluded from the train/test contrast; full N has no10%gap. Raw time-course correlations share monotone drift and are not causal.

## Code revision

unknown; current code at `ec5b452481b89b6901a07f4cb660c66eb6d06e29`. Registry published by `f426cb6c9179e76eaa72202ea750271da1d28a1f`; this does not establish which revision produced historical runs. Source reports may name pre-refactor paths; current entry points are under `scripts/`, reusable modules under `src/latentdyn/`, frozen analyses under `archive/`.
