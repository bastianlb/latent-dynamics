# Re-analysis of the v1 study (81 runs) against the external review — 2026-10-15

Scripts: `/homes/lbastian/latent-dynamics/reanalysis/` (`run_all.sh` reruns everything).
Outputs: this directory (CSVs, PNGs, `issue*_summary.txt`). New metric options in `metrics.py`:
`offline_linear_separability`, `pmag_fixed`, `fixed_scale_grid` (defaults unchanged).

## 1. Convergence timing — FAILS as a general claim (holds on MNIST only)
Baseline-normalised progress p(t) = (x_t − x_0)/(x_T − x_0), D=16 probe/none, mean of 3 seeds.
- MNIST: acc t90 = 4.3%, Fisher 22% (paper's numbers reproduce; normalisation barely matters), H1 lifetime-sum 16%. PMag slow and non-monotone.
- Fashion-MNIST: Fisher, H1 and raw norm already ≥90% of their change at step 50, before accuracy (5.7%). Order reversed.
- CIFAR-10: Fisher t90 = 21%, accuracy 41%. Order reversed.
- First 50-step interval holds 87% of the accuracy change and 63% of the Fisher change (MNIST). On CIFAR-10, H1 and PMag overshoot 2–3× their final change inside it.
- Dense CPU replay (steps 0–200): H1 L1 leads accuracy on MNIST (p = 0.54 vs 0.38 at step 3). On CIFAR-10, H1 reaches 4.5× its final change at steps 12–24, a transient invisible at 50-step logging.
- Files: `timing_per_run.csv`, `timing_summary.csv`, `dense_early.csv`, `fig_timing_progress.png`, `fig_dense_early.png`.

## 2. Topology redundant with geometry — reproduced; largely true for the H1 lifetime-sum
- ρ(Fisher, H1 lifetime-sum) across runs: MNIST −0.912 (−0.875 excl. AE); Fashion −0.79/−0.71; CIFAR −0.73/−0.88.
- Within-run, first-differenced: median ρ ≈ −0.3 (non-AE). Much of the between-run coupling is shared trend/config effects.
- Partial ρ controlling log raw norm and D: −0.78 (MNIST), −0.50 (Fashion), −0.84 (CIFAR).
- Disagreements exist: equal-Fisher configs differ 4–6.6× in H1 (MNIST BN D16 vs D2: 10.0 vs 1.7; Fashion D16 vs D2: 8.1 vs 1.2). D=2→4 raises both.
- H1 noise: 66–73% of H1 lifetime at D=16 is from the bottom-90% of bars; bars < 0.05 mean-pairwise-distance carry 63–83%.
- Top-k bars: MNIST top-5/10 H1 independent of Fisher (ρ ≈ −0.05, partial ≈ 0); Fashion/CIFAR moderately coupled (ρ −0.4 to −0.7).
- Files: `corr_final_*.csv`, `within_run_corr.csv`, `h1_noise_final.csv`, `fig_corr_matrices_final.png`, `fig_h1_noise.png`.

## 3. PMag scale window — FAILS
- The ~6× ℓ2 PMag boost (MNIST 6.54×) is a window artifact: 0.56× under a dataset-shared window, 0.34–0.39× under wide/universal windows.
- Across the 27 MNIST runs, rank corr between original and shared-window PMag = −0.93 (sign flip). Original PMag ~ +0.96 with Fisher; shared-window PMag ~ −0.95.
- Trajectory: baseline PMag rises (final/initial 1.2, t90 52%); under a fixed window it falls (0.66–0.72, t90 2%).
- Validation: re-integrating cached fixed-grid Mag(t) with the original per-snapshot window matches `timeseries.csv` to 0.005%.
- Files: `pmag_windows_final.csv`, `pmag_l2_ratio.csv`, `pmag_trajectory_shape.csv`, `fig_pmag_windows_trajectories.png`.

## 4. Generalisation gap — "decoupling" FAILS (unsupported)
- v1 snapshot `train_acc` is a running train-mode mean over the last 50 batches, not eval-mode accuracy. This causes the negative early gaps (Fig. 2) and CIFAR gaps to −3.4%. Eval-mode full-train accuracy recomputed from every `model_final.pt` → `final_train_eval.csv` (test acc reproduces exactly).
- MNIST gap 0.7–1.4%, SD 0.19 pts, seed noise same size. n=8 configs: 80% power only for |r| ≥ 0.85; config-level bootstrap CIs ≈ ±0.7 (e.g. MNIST Fisher r = −0.03 [−0.63, +0.85]). "|r| < 0.1" cannot be concluded.
- Fashion r = +0.37 reproduces (run level p = 0.08); config-level H1 +0.59 (p = 0.12, CI [−0.44, +0.99]).
- CIFAR with eval-mode gap: config-level r(gap, Fisher) = −0.79, r(gap, H1) = +0.81 (both p = 0.02), confounded with capacity (r(gap, log2 D) = +0.73), uncorrected — exploratory.
- Files: `gap_stats.csv`, `gap_ranges.csv`, `final_train_eval.csv`, `fig_gap_config_level.png`.

## 5. Factual consistency
- Fig. 2 caption wrong: panels are left/right; left = gap vs step (no geometry, gap climbs −3%→+1%, not "stable"); right = final gap vs final H1 lifetime-sum. No Fisher panel. No generating script existed (recovered from an earlier subagent transcript).
- Capacity saturation fails on CIFAR-10: acc by D 2/4/8/16/32 = .406/.573/.650/.670/.674 (saturates at D=16). MNIST/Fashion saturate at D=4. Fisher argmax D = 4/4/2.
- "Fine temporal resolution": weakened (see 1).
- torch 2.14.1+cu130 installed (paper consistent). Style file `icml2025.sty` with `[accepted]`.

## 6. Autoencoder accuracy — final value SURVIVES, trajectory biased
- Online AE probe within 0.9 pts of an offline probe on final full-train latents (MNIST .878 vs .884). Across 81 runs online is 0.7 pts below offline (SD 0.5).
- Online AE probe lags offline by 24 pts over the first 10 snapshots (MNIST); AE t90 step 517 online vs 117 offline.
- Offline accuracy uncorrelated with Fisher/H1/PMag across runs (|ρ| ≤ 0.18) on MNIST and Fashion.
- Files: `offline_vs_online_final.csv`, `offline_vs_online_traj.csv`, `final_offline_probe_full.csv`, `fig_offline_probe.png`.

## Claims to drop or rephrase
1. "Accuracy before geometry on every dataset" → MNIST only.
2. "Accuracy saturates by low capacity on all three datasets" → drop.
3. "~6× ℓ2 PMag" and cross-run PMag levels → drop; use fixed window or Mag(t) at fixed t.
4. "Decoupled from the generalisation gap" → drop everywhere.
5. "Fine temporal resolution" and independence of topology → rephrase; H1 lifetime-sum is mostly a short-bar/dimension effect mirroring Fisher.
6. Fix Fig. 2 caption, style file/[accepted]; use offline probe accuracy for AE timing.

## Positive / surprising
- MNIST long H1 bars (top-5/10) carry information independent of Fisher.
- Mag(t=1) nearly uncorrelated with Fisher on Fashion/CIFAR (ρ +0.06, +0.01) — candidate non-redundant magnitude statistic.
- Fisher ≈ duplicate of separation_mean and compactness (|ρ| = 0.99).
- The early H1/PMag transient on CIFAR-10 is only visible with dense logging → v2 must log densely early and use eval-mode train accuracy (`--train_eval full`).
