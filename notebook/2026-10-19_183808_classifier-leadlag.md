# Classifier dense latent lead-lag

**Date**: 2026-10-19 18:38:08 UTC
**Status**: back-filled 2026-10-26 from existing artifacts
**Datasets**: cifar10, fashion_mnist, mnist
**Tags**: back-fill, classifier-leadlag

Date source: earliest verified config/report artifact mtime in the inspected scope: `/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent/_run_artifacts/runs/mnist_probe_none_d16_s0_dl_adam_nfull_wd0/config.json` = `2026-10-19T18:38:08.608794+00:00`. This is filesystem evidence, not a recovered launch timestamp. See [config provenance](data/config-provenance.csv) and [artifact provenance](data/artifact-provenance.csv).

## Goal

Test whether position-unpredictable displacement averages out with lag and whether spikes have precursors.

## Method

Per-step dense latents, Adam/SGD, multiple N/wd/noise settings; lag1–1000, training/test/logit frames, rounding-floor exclusions. Six MNIST +six FM +two CIFAR source runs, with checkpoints for NTK replay.

Historical execution was on coeus. Exact historical GPU/PID and launch environment are unknown unless a preserved source says otherwise; no launches were made for this back-fill. Registry commands are reconstructed reproductions, not launch logs.

## Runs

**Count**: 14 registry rows.

[Registry](../results/runs.csv), [sweep index](../results/sweeps.csv): filter `group in ['dynamics_v2/dense_latent/_run_artifacts']`. Counts describe registered runs, not a fresh completion audit.

| Registry group | Count | Registry status |
|---|---:|---|
| `dynamics_v2/dense_latent/_run_artifacts` | 14 | complete: 14 |

Resolved configs are `/vol/bitbucket/lbastian/latent-dynamics/<rel_path>/config.json`, using each selected registry `rel_path`. Commands, datasets, seeds and overrides are in the same rows; `results/config_defaults.json` records omitted defaults. All 14 selected config paths were stat-checked; their combined file size is 24116 bytes. Example config: `/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent/_run_artifacts/runs/cifar10_probe_none_d512_s0_dl_rn18_n10000_ln0/config.json`.

CIFAR dense files were described as ~20 GB each in the old digest; bulk sizes were not remeasured.

Absolute coeus artifacts (retained remotely):

- `/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent`
- `/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent/analysis`

No recursive size walk was used. Where a bulk size is unavailable, it remains unmeasured; source-file sizes below and provenance CSVs are exact metadata observations.

## Results

dynamics_report/report.tex: MNIST unexplained residual share .71–.93 (median .78) across lags; energy scales ~lag^2 through ~10 steps then ~lag. FM residual is U-shaped, .53–.66 near lag10. CIFAR class-wise share ~0 at lag1, .18–.42 by lag1000, residual ~95% unexplained. Lines stop where fp16 rounding dominates; one seed per dense condition does not provide an error bar.

Unless explicitly called a direct metadata/table check, headline numbers below are copied from the cited reports and **not re-verified against raw data**. Original reports can contain superseded recommendations; this entry records the later controls and caveats.

Sources (links are local preserved copies; absolute paths identify the originals on coeus):

| Source | Absolute coeus path | Bytes | mtime UTC |
|---|---|---:|---|
| [dynamics_report/report.tex](data/dynamics_report/report.tex) | `/vol/bitbucket/lbastian/latent-dynamics/dynamics_report/report.tex` | 41432 | 2026-10-26T18:35:37.302217+00:00 |

## Conclusion

The residual never averages out; temporal coherence is not a demonstrated generalization early warning.

## Next Steps

Keep rounding-floor exclusions and separate dataset/lag effects. Do not infer a latent-only causal precursor from coherent motion.

## Code revision

unknown; current code at `ec5b452481b89b6901a07f4cb660c66eb6d06e29`. Registry published by `f426cb6c9179e76eaa72202ea750271da1d28a1f`; this does not establish which revision produced historical runs. Source reports may name pre-refactor paths; current entry points are under `scripts/`, reusable modules under `src/latentdyn/`, frozen analyses under `archive/`.
