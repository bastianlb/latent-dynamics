# Null / power test of the train-minus-test affine-residual signature

Runs (all CPU, wd0): NULL N=60000 (every 4 steps to step 7000 = epoch 15) and N=4000 (every 4 to step 9000 = epoch 281), seeds 0-2; POWER N=1000 and N=250 seeds 3-5 added (now 6 seeds each). Launcher run_ae_dense_null.py, analysis ae_dense_null.py (ae_dense_leadlag.py and its outputs untouched). Storage written: about 4 GB on /vol/bitbucket (dense_ae.npz 0.15-0.35 GB per run).

## Verdict
**(a) in the window where the signature was claimed to be early (epoch 6-14): it is NOT an overfitting indicator.** There aff_diff is 0.0036 +- 0.0034 at full N (gap -0.001, i.e. none), 0.0079 +- 0.0063 at N=4000, 0.0048 +- 0.0030 at N=1000, 0.0001 +- 0.0036 at N=250 (seed mean +- sd). Full N, which never overfits, is indistinguishable from N=1000; N=250, with the largest eventual gap, is the lowest. No ordering with eventual gap. Per-snapshot amplitude (0.002-0.008) is below the split-half test-vs-test sd of a single snapshot (0.009-0.011), so a single-run reading of the early signature is within the noise floor; only window/seed averaging shows a small positive offset (~0.003) that is also present at full N (= 'train points are being trained on'/subset offset).
**The signature does scale with the gap later**, but only once the gap is open: window-mean aff_diff at epoch 30-60: 0.008 (N=4000) < 0.017 (N=1000) ~ 0.019 (N=250); at 60-120: 0.009 < 0.019 < 0.022; at 120-250 0.011 < 0.025, 0.016 (N=250 declines while the gap keeps growing). These ordered values coincide with or follow the gap (rel gap >10% at epoch 50/32/41 for N=4000/1000/250), so it carries no lead. The earlier 'epoch 6-14 lead' came from broken-stick fits to a slow drift; the drift is not N-dependent at that stage.
Caveat: full N covers only epochs 0-15 (7000 steps); a full-N reading at epoch 30+ would need ~14000+ steps (~30 min CPU per run). Per-point exposure: comparing at equal epochs (equal exposure) full and N=1000 agree, comparing at equal steps full N is lower than N=1000/250 (0.006 vs 0.022-0.037 at step 1000-4000) because small N sits at higher epochs.


Definitions in ae_dense_null.py docstring. aff_diff = adj. residual share of global affine fit of latent velocity (train probe) minus (test probe); W = 12 steps for all runs (NOT identical to the W=6 steps of RESULTS.md, so numbers differ slightly). All entries seed mean +- sd (n=3 for full/4000, n=6 for 1000/250 = seeds 0-5). Epoch = step*128/N.

## aff_diff at fixed EPOCH of per-point exposure
| N | ep 2 | ep 4 | ep 6 | ep 10 | ep 14 | peak | peak epoch | first crossing of floor+3sd (epoch) |
|---|---|---|---|---|---|---|---|---|
| full | 0.0002 +- 0.0068 | 0.0031 +- 0.0057 | 0.0011 +- 0.0028 | 0.0043 +- 0.0019 | 0.0073 +- 0.0008 | 0.0199 +- 0.0047 | 6.3 +- 5.1 | n/a |
| 4000 | 0.0040 +- 0.0055 | 0.0014 +- 0.0028 | 0.0056 +- 0.0034 | 0.0113 +- 0.0097 | 0.0059 +- 0.0035 | 0.0252 +- 0.0047 | 222.9 +- 44.9 | 260.7 +- 0.0 |
| 1000 | 0.0006 +- 0.0016 | 0.0027 +- 0.0018 | 0.0024 +- 0.0017 | 0.0042 +- 0.0030 | 0.0079 +- 0.0059 | 0.0463 +- 0.0031 | 346.0 +- 93.6 | 114.6 +- 76.1 |
| 250 | -0.0009 +- 0.0022 | -0.0009 +- 0.0022 | -0.0009 +- 0.0022 | 0.0002 +- 0.0032 | 0.0034 +- 0.0042 | 0.0919 +- 0.0075 | 943.6 +- 79.3 | 186.2 +- 153.8 |

## aff_diff at fixed STEP
| N | step 200 | step 500 | step 1000 | step 2000 | step 4000 |
|---|---|---|---|---|---|
| full | 0.0006 +- 0.0020 | 0.0026 +- 0.0027 | 0.0054 +- 0.0120 | 0.0008 +- 0.0034 | 0.0059 +- 0.0096 |
| 4000 | 0.0050 +- 0.0055 | 0.0070 +- 0.0098 | 0.0063 +- 0.0054 | 0.0083 +- 0.0055 | 0.0072 +- 0.0067 |
| 1000 | 0.0109 +- 0.0054 | 0.0169 +- 0.0058 | 0.0220 +- 0.0028 | 0.0298 +- 0.0060 | 0.0369 +- 0.0070 |
| 250 | 0.0199 +- 0.0026 | 0.0164 +- 0.0034 | 0.0205 +- 0.0400 | 0.0642 +- 0.0259 | n/a |

## Noise floors and gap
| N | split-half test-test sd (median over time) | split-half train-train sd | median bias test-test | threshold floor+3sd | recon gap at epoch 6 | gap at end of window | rel gap>10% at epoch |
|---|---|---|---|---|---|---|---|
| full | 0.0105 +- 0.0004 | 0.0107 +- 0.0005 | -0.0010 +- 0.0007 | 0.0305 +- 0.0012 | -0.0016 +- 0.0006 | -0.0005 +- 0.0006 | n/a |
| 4000 | 0.0104 +- 0.0003 | 0.0107 +- 0.0002 | -0.0006 +- 0.0010 | 0.0304 +- 0.0017 | -0.0021 +- 0.0005 | 0.0489 +- 0.0027 | 49.6 +- 5.5 |
| 1000 | 0.0101 +- 0.0005 | 0.0115 +- 0.0012 | -0.0004 +- 0.0017 | 0.0299 +- 0.0027 | 0.0027 +- 0.0040 | 0.1891 +- 0.0034 | 32.0 +- 2.1 |
| 250 | 0.0089 +- 0.0008 | 0.0153 +- 0.0011 | 0.0002 +- 0.0009 | 0.0269 +- 0.0028 | 0.0073 +- 0.0095 | 0.5034 +- 0.0145 | 41.3 +- 1.9 |

## Train-only and test-only residual share (epoch 6)
| N | aff_tr | aff_te |
|---|---|---|
| full | 0.3324 +- 0.0246 | 0.3258 +- 0.0150 |
| 4000 | 0.1889 +- 0.0362 | 0.1827 +- 0.0352 |
| 1000 | 0.0610 +- 0.0193 | 0.0591 +- 0.0202 |
| 250 | 0.0287 +- 0.0123 | 0.0296 +- 0.0124 |

## Figures
fig_null_affdiff.png (aff_diff vs step / epoch, gap, train and test parts, z-score), fig_null_floor.png (split-half floor), fig_null_vs_gap.png (amplitude vs gap)
## Window-averaged aff_diff (mean over snapshots in the epoch window) vs floor
Floor column: sd across all runs/seeds (all N pooled) of the same window-average of the test-vs-test split-half difference (no train-test asymmetry, so any spread is pure noise); 'z' = (seed-mean aff_diff - mean floor) / floor-sd.

| window (epoch) | floor sd (windowed) | full | 4000 | 1000 | 250 |
|---|---|---|---|---|---|
| 2-6 | 0.0011 | 0.0028 +- 0.0035 (z=3.7; mean gap -0.002; n=3) | 0.0038 +- 0.0023 (z=3.3; mean gap -0.004; n=3) | 0.0024 +- 0.0014 (z=2.4; mean gap 0.003; n=6) | n/a |
| 6-14 | 0.0011 | 0.0036 +- 0.0034 (z=3.9; mean gap -0.001; n=3) | 0.0079 +- 0.0063 (z=7.3; mean gap 0.001; n=3) | 0.0048 +- 0.0030 (z=4.2; mean gap 0.005; n=6) | 0.0001 +- 0.0036 (z=0.1; mean gap 0.006; n=6) |
| 14-30 | 0.0012 | 0.0033 +- 0.0039 (z=3.7; mean gap -0.001; n=3) | 0.0095 +- 0.0050 (z=8.1; mean gap 0.008; n=3) | 0.0099 +- 0.0037 (z=7.9; mean gap 0.017; n=6) | 0.0083 +- 0.0028 (z=6.5; mean gap 0.015; n=6) |
| 30-60 | 0.0017 | n/a | 0.0084 +- 0.0035 (z=4.9; mean gap 0.017; n=3) | 0.0167 +- 0.0044 (z=10.1; mean gap 0.039; n=6) | 0.0190 +- 0.0048 (z=11.0; mean gap 0.047; n=6) |
| 60-120 | 0.0009 | n/a | 0.0094 +- 0.0033 (z=11.0; mean gap 0.027; n=3) | 0.0189 +- 0.0035 (z=20.9; mean gap 0.069; n=6) | 0.0217 +- 0.0042 (z=23.2; mean gap 0.120; n=6) |
| 120-250 | 0.0013 | n/a | 0.0111 +- 0.0040 (z=9.2; mean gap 0.040; n=3) | 0.0250 +- 0.0024 (z=19.8; mean gap 0.113; n=6) | 0.0155 +- 0.0021 (z=11.8; mean gap 0.242; n=6) |

fig_null_windows.png (window means vs epoch with floor band; aff_diff vs gap)
