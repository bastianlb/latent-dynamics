# v2 Fashion-MNIST and CIFAR-10 replication

**Date**: 2026-10-15 10:29:28 UTC
**Status**: back-filled 2026-10-26 from existing artifacts
**Datasets**: cifar10, fashion_mnist
**Tags**: back-fill, v2-dataset-replication

Date source: earliest verified config/report artifact mtime in the inspected scope: `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs/cifar10_probe_none_d512_s2_rn18_nfull_ln0/config.json` = `2026-10-15T10:29:28.199187+00:00`. This is filesystem evidence, not a recovered launch timestamp. See [config provenance](data/config-provenance.csv) and [artifact provenance](data/artifact-provenance.csv).

## Goal

Check whether MNIST phase, geometry and forecasting observations transfer across datasets.

## Method

105 FM MLP runs mirror phase/gap grids (35 configs, three seeds). 18 CIFAR-10 ResNet-18 d512 runs vary N=10000/full and label noise 0/.2/.4, three seeds, augmentation. Launchers jobs_FM.txt and cifar_block.sh.

Historical execution was on coeus. Exact historical GPU/PID and launch environment are unknown unless a preserved source says otherwise; no launches were made for this back-fill. Registry commands are reconstructed reproductions, not launch logs.

## Runs

**Count**: 123 registry rows.

[Registry](../results/runs.csv), [sweep index](../results/sweeps.csv): filter `group in ['runs_v2/FM', 'runs_v2/cifar_block']`. Counts describe registered runs, not a fresh completion audit.

| Registry group | Count | Registry status |
|---|---:|---|
| `runs_v2/FM` | 105 | complete: 105 |
| `runs_v2/cifar_block` | 18 | complete: 18 |

Resolved configs are `/vol/bitbucket/lbastian/latent-dynamics/<rel_path>/config.json`, using each selected registry `rel_path`. Commands, datasets, seeds and overrides are in the same rows; `results/config_defaults.json` records omitted defaults. All 123 selected config paths were stat-checked; their combined file size is 127530 bytes. Example config: `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs/fashion_mnist_probe_none_d16_s0_gap_n1000_ln0.2/config.json`.



Absolute coeus artifacts (retained remotely):

- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs`
- `/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/ext2/fashion`
- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/jobs_FM.txt`
- `/vol/bitbucket/lbastian/latent-dynamics/runs_v2/cifar_block.sh`

No recursive size walk was used. Where a bulk size is unavailable, it remains unmeasured; source-file sizes below and provenance CSVs are exact metadata observations.

## Results

analysis_v2/REPORT.md gives CIFAR gaps .16/.33/.51 for N=10000 at noise 0/.2/.4. FM exploratory forecasting gives 0/10 ridge equivalence cells at the stated margins; the GBM gap-at-1000 hint +.254 [.034,.546] does not establish a general incremental geometry signal. These are six CIFAR configs and 35 FM configs, not independent timepoints.

Unless explicitly called a direct metadata/table check, headline numbers below are copied from the cited reports and **not re-verified against raw data**. Original reports can contain superseded recommendations; this entry records the later controls and caveats.

Sources (links are local preserved copies; absolute paths identify the originals on coeus):

| Source | Absolute coeus path | Bytes | mtime UTC |
|---|---|---:|---|
| [analysis_v2/REPORT.md](data/analysis_v2/REPORT.md) | `/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/REPORT.md` | 23402 | 2026-10-16T22:12:53.490011+00:00 |
| [paper_v2/NUMBERS.md](data/paper_v2/NUMBERS.md) | `/vol/bitbucket/lbastian/latent-dynamics/paper_v2/NUMBERS.md` | 57822 | 2026-10-16T22:26:32.364068+00:00 |

## Conclusion

FM broadly replicates the ridge null; dataset and architecture effects remain intertwined on CIFAR.

## Next Steps

Use the confirmatory FM study for forecasting claims; no confirmatory CIFAR study was undertaken.

## Code revision

unknown; current code at `ec5b452481b89b6901a07f4cb660c66eb6d06e29`. Registry published by `f426cb6c9179e76eaa72202ea750271da1d28a1f`; this does not establish which revision produced historical runs. Source reports may name pre-refactor paths; current entry points are under `scripts/`, reusable modules under `src/latentdyn/`, frozen analyses under `archive/`.
