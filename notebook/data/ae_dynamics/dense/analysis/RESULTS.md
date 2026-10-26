# Dense early-window AE runs: does latent geometry lead or lag generalization?

MNIST AE d=16, same hyperparameters/seeds as the grid, snapshots every 2 steps (N=1000 to step 4000 = epoch 512; N=250 to step 2000 = epoch 1000). Epoch = step*128/N. 3 seeds per config; entries are seed-mean [bootstrap 95% CI over seeds] (n=3: coarse, treat as a spread, not an inferential CI). Definitions: see docstring of ae_dense_leadlag.py.

## Verification vs grid
All 12 dense runs: log-schedule snapshots vs grid files: max |latent diff| = 0.00e+00, max |recon scalar diff| = 0.00e+00; dense_ae.npz entries at grid steps vs grid: latent 0.00e+00, recon 0.00e+00 (see verify.json).

## N1000_wd0
- test-recon minimum t*: epoch 126.5 [119.3, 137.2]; min value 0.2471 [0.2458, 0.2479]; gap(abs) at t* 0.0915 [0.0881, 0.0976]
- test recon within 5% of min at epoch 64.1 [62.2, 66.8]; within 2%: 84.2 [82.7, 87.0]
- gap(abs) t25 27.2 [23.6, 33.8], t50 53.1 [48.6, 58.9]; relative gap te/tr-1 > 10% at epoch 30.5 [29.2, 32.5]
- gap vs test-recon derivative xcorr lag (ln t): -0.18 [-0.28, -0.11]

- gap hinge onset (broken-stick in ln t on [epoch 5, t*]): epoch 15.7 [12.8, 20.1], R2 0.99 [0.98, 0.99]

| signature | type | hinge onset (ep) | hinge R2 | hinge detectable | ln(tau/tau_gap) | ln(tau/t*) | value at t* (ref pixel-PCA) | t50 (ep) [frac. of t* value] | ln(t50/gap t50) | xcorr lag vs gap (ln t) | xcorr peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spread_lab | latent-only | 19.9 [5.4, 33.8] | 0.68 [0.51, 0.96] | 3/3 | 0.01 [-1.31, 0.97] | -2.09 [-3.23, -1.29] | 1.073 [1.065, 1.082] (1.060) | 55.0 [55.0, 55.0] (1/3) | 0.06 [0.06, 0.06] | 0.16 [-0.48, 0.69] | 0.24 |
| var_ratio | latent-only | 5.4 [5.4, 5.4] | 0.83 [0.79, 0.93] | 3/3 | -1.04 [-1.31, -0.86] | -3.15 [-3.23, -3.09] | 1.031 [1.021, 1.042] (1.028) | 13.6 [13.6, 13.6] (1/3) | -1.47 [-1.47, -1.47] | -0.17 [-0.79, 0.79] | 0.10 |
| effrank | latent-only | 10.2 [9.2, 11.4] | 0.73 [0.61, 0.89] | 3/3 | -0.41 [-0.78, -0.21] | -2.52 [-2.70, -2.35] | 1.055 [1.035, 1.072] (1.044) | n/a (0/3) | n/a | 0.15 [-0.44, 0.62] | 0.21 |
| nn_ratio | latent-only | 17.0 [6.6, 32.3] | 0.55 [0.35, 0.79] | 2/3 | 0.39 [-0.05, 0.83] | -1.81 [-2.32, -1.31] | 0.986 [0.967, 1.002] (1.002) | n/a (0/3) | n/a | 0.41 [0.10, 0.57] | 0.17 |
| vel_ratio | latent-only | 11.3 [5.7, 14.3] | 0.13 [0.05, 0.23] | 0/3 | n/a | n/a | 0.995 [0.994, 0.997] (-) | n/a (0/3) | n/a | 0.18 [-0.64, 0.79] | 0.13 |
| aff_diff | latent-only | 5.8 [5.4, 6.6] | 0.69 [0.63, 0.80] | 3/3 | -0.98 [-1.11, -0.86] | -3.08 [-3.12, -3.03] | 0.019 [0.018, 0.019] (-) | 118.6 [110.3, 134.7] (3/3) | 0.80 [0.76, 0.83] | -0.07 [-0.35, 0.28] | 0.08 |
| dec_lab | DECODER-tied | 22.2 [21.7, 22.6] | 0.92 [0.83, 0.97] | 3/3 | 0.37 [0.11, 0.57] | -1.74 [-1.81, -1.70] | 0.119 [0.101, 0.147] (-) | 65.7 [60.9, 70.4] (2/3) | 0.20 [0.18, 0.23] | 0.07 [-0.04, 0.21] | 0.81 |
| dec_km | DECODER-tied | 36.3 [14.9, 77.9] | 0.70 [0.20, 0.98] | 2/3 | -0.08 [-0.22, 0.05] | -2.11 [-2.14, -2.08] | 0.138 [0.065, 0.176] (-) | 63.2 [60.2, 66.3] (2/3) | 0.14 [0.12, 0.15] | 0.30 [-0.02, 0.79] | 0.83 |

## N1000_wd1e-4
- test-recon minimum t*: epoch 144.0 [139.5, 151.0]; min value 0.2442 [0.2385, 0.2491]; gap(abs) at t* 0.0999 [0.0967, 0.1032]
- test recon within 5% of min at epoch 69.7 [66.0, 72.4]; within 2%: 92.4 [88.3, 96.8]
- gap(abs) t25 28.9 [25.6, 32.3], t50 56.2 [51.7, 58.6]; relative gap te/tr-1 > 10% at epoch 30.2 [28.4, 31.5]
- gap vs test-recon derivative xcorr lag (ln t): -0.45 [-0.59, -0.33]

- gap hinge onset (broken-stick in ln t on [epoch 5, t*]): epoch 16.0 [14.1, 18.6], R2 0.99 [0.99, 0.99]

| signature | type | hinge onset (ep) | hinge R2 | hinge detectable | ln(tau/tau_gap) | ln(tau/t*) | value at t* (ref pixel-PCA) | t50 (ep) [frac. of t* value] | ln(t50/gap t50) | xcorr lag vs gap (ln t) | xcorr peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spread_lab | latent-only | 25.7 [11.0, 52.4] | 0.68 [0.19, 0.95] | 2/3 | -0.18 [-0.25, -0.11] | -2.48 [-2.56, -2.41] | 1.076 [1.060, 1.086] (1.060) | n/a (0/3) | n/a | 0.16 [-0.67, 0.74] | 0.29 |
| var_ratio | latent-only | 5.4 [5.4, 5.4] | 0.84 [0.72, 0.94] | 3/3 | -1.07 [-1.23, -0.95] | -3.27 [-3.32, -3.24] | 1.035 [1.019, 1.043] (1.028) | n/a (0/3) | n/a | -0.68 [-0.79, -0.50] | 0.20 |
| effrank | latent-only | 6.9 [5.4, 7.9] | 0.43 [0.28, 0.62] | 1/3 | -0.66 [-0.66, -0.66] | -2.95 [-2.95, -2.95] | 1.046 [1.036, 1.064] (1.044) | n/a (0/3) | n/a | 0.07 [-0.76, 0.74] | 0.10 |
| nn_ratio | latent-only | 31.0 [5.4, 53.9] | 0.39 [0.06, 0.62] | 1/3 | -1.03 [-1.03, -1.03] | -3.32 [-3.32, -3.32] | 0.976 [0.957, 0.994] (1.002) | n/a (0/3) | n/a | 0.56 [0.33, 0.75] | 0.08 |
| vel_ratio | latent-only | 7.3 [6.4, 9.1] | 0.12 [0.04, 0.17] | 0/3 | n/a | n/a | 0.997 [0.993, 1.004] (-) | n/a (0/3) | n/a | 0.64 [0.43, 0.76] | 0.11 |
| aff_diff | latent-only | 7.2 [5.6, 9.3] | 0.79 [0.74, 0.81] | 3/3 | -0.82 [-1.03, -0.42] | -3.02 [-3.29, -2.72] | 0.027 [0.018, 0.033] (-) | 109.2 [90.4, 138.2] (3/3) | 0.65 [0.43, 0.98] | -0.34 [-0.48, -0.08] | 0.14 |
| dec_lab | DECODER-tied | 33.0 [26.8, 45.4] | 0.84 [0.80, 0.90] | 3/3 | 0.70 [0.36, 1.09] | -1.50 [-1.66, -1.20] | 0.101 [0.076, 0.136] (-) | 79.1 [75.5, 82.7] (2/3) | 0.36 [0.35, 0.38] | 0.64 [0.35, 0.79] | 0.77 |
| dec_km | DECODER-tied | 27.1 [11.5, 54.4] | 0.72 [0.54, 0.90] | 3/3 | 0.29 [-0.29, 1.35] | -1.91 [-2.58, -0.95] | 0.118 [0.107, 0.133] (-) | 92.4 [73.0, 108.5] (3/3) | 0.49 [0.22, 0.74] | 0.62 [0.28, 0.79] | 0.80 |

## N250_wd0
- test-recon minimum t*: epoch 119.5 [114.7, 123.9]; min value 0.3438 [0.3385, 0.3491]; gap(abs) at t* 0.1616 [0.1585, 0.1637]
- test recon within 5% of min at epoch 79.5 [74.8, 82.9]; within 2%: 92.5 [87.0, 97.3]
- gap(abs) t25 43.0 [38.9, 48.1], t50 66.9 [65.5, 68.6]; relative gap te/tr-1 > 10% at epoch 40.3 [38.9, 43.0]
- gap vs test-recon derivative xcorr lag (ln t): -0.72 [-0.80, -0.55]

- gap hinge onset (broken-stick in ln t on [epoch 5, t*]): epoch 26.0 [22.6, 29.1], R2 0.98 [0.98, 0.99]

| signature | type | hinge onset (ep) | hinge R2 | hinge detectable | ln(tau/tau_gap) | ln(tau/t*) | value at t* (ref pixel-PCA) | t50 (ep) [frac. of t* value] | ln(t50/gap t50) | xcorr lag vs gap (ln t) | xcorr peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spread_lab | latent-only | 33.0 [5.4, 71.4] | 0.53 [0.03, 0.82] | 2/3 | -0.85 [-1.43, -0.27] | -2.41 [-3.13, -1.68] | 1.173 [1.148, 1.198] (1.185) | 31.7 [13.3, 50.2] (2/3) | -0.94 [-1.61, -0.27] | -0.68 [-0.80, -0.49] | -0.07 |
| var_ratio | latent-only | 10.2 [5.4, 19.7] | 0.87 [0.81, 0.98] | 3/3 | -1.13 [-1.68, -0.29] | -2.66 [-3.13, -1.76] | 1.128 [1.095, 1.150] (1.135) | 12.3 [12.3, 12.3] (1/3) | -1.69 [-1.69, -1.69] | -0.80 [-0.80, -0.80] | 0.04 |
| effrank | latent-only | 16.9 [5.4, 30.7] | 0.50 [0.20, 0.74] | 2/3 | -1.01 [-1.58, -0.43] | -2.59 [-3.05, -2.13] | 1.099 [1.086, 1.113] (1.123) | 48.1 [48.1, 48.1] (1/3) | -0.31 [-0.31, -0.31] | -0.74 [-0.80, -0.63] | 0.06 |
| nn_ratio | latent-only | 31.4 [17.7, 53.7] | 0.48 [0.30, 0.80] | 1/3 | -0.39 [-0.39, -0.39] | -1.87 [-1.87, -1.87] | 0.943 [0.913, 1.000] (0.974) | 65.5 [65.5, 65.5] (1/3) | -0.02 [-0.02, -0.02] | 0.12 [-0.80, 0.80] | 0.05 |
| vel_ratio | latent-only | 26.4 [5.4, 68.4] | 0.20 [0.08, 0.31] | 0/3 | n/a | n/a | 1.025 [1.019, 1.036] (-) | 82.9 [37.9, 118.8] (3/3) | 0.11 [-0.55, 0.58] | -0.51 [-0.78, -0.34] | -0.09 |
| aff_diff | latent-only | 9.7 [6.9, 11.9] | 0.70 [0.64, 0.81] | 3/3 | -1.00 [-1.44, -0.78] | -2.53 [-2.86, -2.26] | 0.026 [0.023, 0.031] (-) | 37.9 [34.8, 39.9] (3/3) | -0.57 [-0.68, -0.51] | 0.59 [0.32, 0.78] | 0.14 |
| dec_lab | DECODER-tied | 57.8 [52.1, 61.5] | 0.87 [0.77, 0.95] | 3/3 | 0.80 [0.68, 0.97] | -0.73 [-0.79, -0.67] | 0.318 [0.256, 0.381] (-) | 89.8 [85.0, 96.3] (3/3) | 0.29 [0.25, 0.37] | 0.06 [-0.19, 0.20] | 0.86 |
| dec_km | DECODER-tied | 58.6 [36.0, 78.3] | 0.72 [0.28, 0.95] | 2/3 | 0.53 [0.32, 0.75] | -0.91 [-1.16, -0.67] | 0.340 [0.214, 0.427] (-) | 83.5 [77.8, 89.1] (2/3) | 0.22 [0.17, 0.26] | 0.04 [-0.15, 0.32] | 0.84 |

## N250_wd1e-4
- test-recon minimum t*: epoch 124.9 [116.7, 130.0]; min value 0.3450 [0.3367, 0.3518]; gap(abs) at t* 0.1722 [0.1681, 0.1802]
- test recon within 5% of min at epoch 84.0 [81.9, 85.0]; within 2%: 97.6 [93.2, 100.4]
- gap(abs) t25 47.8 [44.0, 53.2], t50 73.0 [68.6, 75.8]; relative gap te/tr-1 > 10% at epoch 44.0 [41.0, 46.1]
- gap vs test-recon derivative xcorr lag (ln t): -0.72 [-0.80, -0.56]

- gap hinge onset (broken-stick in ln t on [epoch 5, t*]): epoch 29.6 [27.8, 33.1], R2 0.98 [0.97, 0.99]

| signature | type | hinge onset (ep) | hinge R2 | hinge detectable | ln(tau/tau_gap) | ln(tau/t*) | value at t* (ref pixel-PCA) | t50 (ep) [frac. of t* value] | ln(t50/gap t50) | xcorr lag vs gap (ln t) | xcorr peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spread_lab | latent-only | 25.9 [5.4, 67.0] | 0.53 [0.12, 0.75] | 2/3 | -1.72 [-1.81, -1.63] | -3.17 [-3.18, -3.16] | 1.164 [1.146, 1.184] (1.185) | 42.0 [14.3, 69.6] (2/3) | -0.82 [-1.65, 0.01] | -0.74 [-0.80, -0.63] | -0.00 |
| var_ratio | latent-only | 10.6 [5.4, 20.9] | 0.90 [0.79, 0.98] | 3/3 | -1.24 [-1.81, -0.29] | -2.69 [-3.18, -1.72] | 1.124 [1.108, 1.145] (1.135) | 11.3 [11.3, 11.3] (1/3) | -1.89 [-1.89, -1.89] | -0.80 [-0.80, -0.80] | 0.18 |
| effrank | latent-only | 17.5 [5.4, 30.5] | 0.48 [0.04, 0.73] | 2/3 | -1.08 [-1.64, -0.52] | -2.56 [-3.07, -2.04] | 1.095 [1.064, 1.138] (1.123) | 38.9 [29.7, 48.1] (2/3) | -0.64 [-0.92, -0.35] | -0.73 [-0.80, -0.60] | 0.08 |
| nn_ratio | latent-only | 27.3 [15.6, 46.7] | 0.36 [0.12, 0.55] | 1/3 | -0.58 [-0.58, -0.58] | -2.01 [-2.01, -2.01] | 0.947 [0.938, 0.956] (0.974) | n/a (0/3) | n/a | 0.08 [-0.80, 0.80] | -0.03 |
| vel_ratio | latent-only | 5.4 [5.4, 5.4] | 0.30 [0.26, 0.35] | 0/3 | n/a | n/a | 1.027 [1.020, 1.036] (-) | 45.7 [9.2, 99.3] (3/3) | -0.90 [-2.11, 0.37] | -0.65 [-0.80, -0.47] | -0.07 |
| aff_diff | latent-only | 14.1 [11.1, 16.6] | 0.79 [0.69, 0.88] | 3/3 | -0.75 [-1.10, -0.52] | -2.19 [-2.46, -2.04] | 0.027 [0.023, 0.033] (-) | 42.7 [41.0, 45.1] (3/3) | -0.54 [-0.58, -0.52] | 0.63 [0.60, 0.67] | 0.22 |
| dec_lab | DECODER-tied | 60.8 [58.1, 65.6] | 0.84 [0.71, 0.93] | 3/3 | 0.72 [0.68, 0.74] | -0.72 [-0.79, -0.68] | 0.336 [0.271, 0.372] (-) | 93.5 [89.1, 99.3] (3/3) | 0.25 [0.20, 0.28] | 0.13 [-0.04, 0.25] | 0.88 |
| dec_km | DECODER-tied | 39.2 [5.4, 62.1] | 0.69 [0.20, 0.94] | 2/3 | 0.61 [0.58, 0.63] | -0.79 [-0.85, -0.74] | 0.342 [0.198, 0.429] (-) | 88.1 [84.0, 92.2] (2/3) | 0.20 [0.20, 0.20] | 0.11 [-0.03, 0.39] | 0.87 |

## Figures
fig_timeseries.png (raw series), fig_progress.png (normalised progress, 1 = value at test minimum), fig_onsets.png (t50 per signature), fig_xcorr.png (cross-correlation lags)

## Verdict (written by hand after inspecting tables and figures)
- Gap timing (all configs): broken-stick onset of the abs gap at epoch ~16 (N=1000) / ~26-30 (N=250); relative gap >10% at epoch 30 (N=1000) / 40-44 (N=250); test recon within 5% of its minimum at epoch 64-84, minimum t* at epoch 120-145. wd=1e-4 shifts all of these by <20% and does not change the minimum location.
- DECODER-tied signature (Spearman error-vs-class-distance, test minus train) LAGS the gap: hinge 22-61 epochs, ln(tau/tau_gap) = +0.4..+0.8. Expected: it is the same event (train error collapsing) seen per-sample.
- Latent-only signatures (spread, total variance, effective rank, nn_ratio, velocity ratio, global-affine residual share) show NO sharp onset. They drift slowly and noisily from epoch ~5 (hinge R2 0.4-0.8, CIs span a factor 3-10, several onsets sit on the window edge at epoch 5.4). Their nominal hinge onsets precede the gap hinge (ln ratio -0.2..-1.7; the global-affine residual-share difference is the most reproducible: epoch 6-14, ln(tau/tau_gap) about -0.8..-1.0, tight across seeds), but (i) the drift starts when train and test latents differ only by the init/sample-subspace offset (var_ratio at t* is 1.03 vs pixel-PCA reference 1.03 at N=1000; 1.13 vs 1.14 at N=250), i.e. excess over the null is ~0; spread_lab excess is +0.01 (N=1000) and -0.01 (N=250); (ii) derivative cross-correlation peaks are 0.0-0.3 (the decoder-tied ones are 0.8), so the lags from cross-correlation are not meaningful for latent-only signals; (iii) n=3 seeds.
- Net: no latent-only signature is a demonstrated early indicator. The only candidate worth a follow-up is the train-minus-test global-affine residual share of the latent velocity field (leads the gap hinge by a factor ~2-3 in time, reproducible across seeds, but small amplitude 0.02-0.03 and possibly tied to train points being optimized directly rather than to memorization). nn_ratio departs from 1 only at epoch ~17-31 (N=1000) / ~27-31 (N=250), i.e. after the gap onset, lag or coincident, and only in 1-2 of 3 seeds.
