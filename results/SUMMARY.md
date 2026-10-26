System of record for experiments: [notebook/](../notebook/README.md), with the live index in [WORKING.md](../notebook/WORKING.md).

# Results summary

Factual summary of the experiments run so far and what they found. Artifact root is
`B = /vol/bitbucket/lbastian/latent-dynamics/`. Per-run hyperparameters and reproduce commands are in
`results/runs.csv` (the per-run index; not duplicated here). Written 2026-10-26 from the files cited in each bullet.
Numbers are copied from those files; where sources disagree the latest is used and the conflict is noted in
"Discrepancies" at the end.

## Overview

The project asks how the latent representation of a small network changes during training, and whether that
dynamics says anything about generalization. A fixed probe set (1000 test, usually 1000 train points) is
snapshotted on a log schedule (about 120 snapshots, 20k steps), and geometric, topological (persistent homology,
magnitude) and vector-field quantities are computed over the trajectory. Models: MLP/CNN/ResNet-18 classifiers
with a latent layer and linear probe (MNIST, Fashion-MNIST, CIFAR-10), and, since 2026-10-21, reconstruction
autoencoders as a label-free control. The study went through five stages: (1) descriptive geometry/topology
(phase 1); (2) a pre-registered test of whether latent geometry forecasts test accuracy or the train/test gap
beyond learning curves and design (null); (3) a dynamical-systems view (slingshot spikes, vector-field
decomposition, tangent-kernel coupling, train vs test); (4) dense per-step logging; (5) an autoencoder replication.
The goal is generalization tied theoretically to the latent space, so open questions are framed that way.
Headline: latent motion is mostly one linear map plus an unpredictable residual set by the tangent kernel and
optimizer state; there are no latent phase transitions; geometry adds no detectable forecasting power for
generalization beyond training design; autoencoder latents record memorization after the fact but do not
anticipate it.

## Experiment blocks

Run counts are approximate (directories or job-file lines); see `results/runs.csv` for exact lists.
Code: `latentdyn.*` modules under `src/latentdyn/` or `scripts/` entry points (thin wrappers); one-off analyses
are frozen under `archive/`. Training is always `latentdyn.train` (`scripts/train.py`); grids are launched by
`latentdyn.sweep` / `latentdyn.launch` or by the shell blocks in `B/runs_v2/*.sh` and `B/runs_v2/jobs_*.txt`.

| Block | Dataset / model | What varied | ~Runs | Artifacts | Producing code |
|---|---|---|---|---|---|
| P1 capacity | MNIST, FM, CIFAR-10; MLP/CNN encoder, linear probe | latent_dim 2,4,8,16,32 x 3 seeds | 15 per dataset | `B/runs/*_probe_none_d*`, `B/sweeps/capacity` | `latentdyn.sweep`, `pipeline.analyze/plot/compare/trajectory` |
| P1 objective | same | probe vs autoencoder vs probe+recon, d16, 3 seeds | 9 per dataset | `B/runs/*_autoencoder_*`, `*_probe_recon_*`, `B/sweeps/contrast` | same |
| P1 normalization | same | none / L2 sphere / BatchNorm, d16, 3 seeds | 9 per dataset | `B/runs/*_probe_l2_*`, `*_probe_bn_*` | same |
| P1 report | all three | 27 runs per dataset (81 + 1 baseline) | 82 | `B/report/report.tex`, `B/report_assets/` | `latentdyn.pipeline.*`; reanalysis in `archive/reanalysis/` -> `B/reanalysis_v2/` |
| v2 phase grid A | MNIST MLP d16, Adam | train_subset {250,1k,4k,16k,full} x wd {0,1e-4,1e-3,1e-2,3e-2} x 3 seeds, 20k steps | 75 | `B/runs_v2/runs` (`*_phase_n*_wd*`) | `B/runs_v2/mnist_block.sh`, `jobs_E2.txt`, `jobs_E.HOLD.txt` |
| v2 gap axis B | MNIST | subset x label_noise {0.2,0.4}, wd 0 | 30 | `*_gap_n*_ln*` | `mnist_block.sh` |
| v2 FM grid | Fashion-MNIST MLP | same grid as A/B | 105 | `*_phase_*`, `*_gap_*` | `jobs_FM.txt` |
| v2 CIFAR | CIFAR-10 ResNet-18, d512 | n {10k,full} x label_noise {0,0.2,0.4}, 3 seeds, augmentation | 18 | `*_rn18_*` | `cifar_block.sh` |
| v2 l2-temperature | MNIST, FM | none / L2 scale 1 / 10 / learned, 5k steps | 24 | `*_l2t_*` | `l2t_block.sh` |
| 40k slingshot (C) | MNIST | N {250,1k,4k} x wd x seeds, 40k steps | 90 | `*_sl40k_*` | `jobs_C.txt` |
| SGD 40k (D) | MNIST, SGD+mom 0.9, lr 0.05 | N x wd | 30 | `*_sgd40k_*` | `jobs_D.txt` |
| Adam eps (F1/F2) | MNIST | adam_eps 1e-8..1e-3, N, wd | 90 + 90 | `*_eps*_n*_wd*` | `jobs_F1.txt`, `jobs_F2.txt`; analysis `archive/slingshot/stability.py` |
| SGD lr (G) | MNIST | SGD lr grid x N x wd | 72 | `*_sgdlr*` | `jobs_G.txt`, `archive/slingshot/sgd_lrB.py` |
| SGD batch/lr (lrB) | MNIST, SGD | batch {32,..} x lr, N {1k,4k} | 48 (+24 matched) | `*_lrB_b*_lr*` | `jobs_lrB.txt`; `B/dynamics_v2/sgd_lrB` |
| Dense per-step (opt) | MNIST Adam | every-step loss and optimizer state, eps x N x wd | 12 | `B/dynamics_v2/dense` (dumps `/data2/lbastian/latent-dynamics/opt_dense`) | `latentdyn.train_aux.optdense`; analysis `archive/dense/*` |
| Dense latent | MNIST (6), FM (6), CIFAR-10 (2) | per-step latents, Adam/SGD, N {1k,4k,full}, wd, noise 0.2 | 14 | `B/dynamics_v2/dense_latent` (CIFAR about 20 GB each) | `latentdyn.train_aux.dense_latent`; `latentdyn.vfield.*`, `latentdyn.ntk.*` |
| Exploratory prediction set | MNIST MLP | 69 configs, n 250..60000, wd, label noise 0.2/0.4, Adam/SGD | 305 | `B/analysis_v2` | `archive/analysis_v2/step2..step9` |
| Confirmatory MNIST | MNIST MLP + small CNN | 28 random off-grid configs (lr, wd, noise, n, arch), seeds 100-102 | 84 | `B/analysis_v2/confirm`, `jobs_CONF.txt` | `archive/analysis_v2/confirm_gen.py`, `step7_confirm.py` |
| Confirmatory FM | Fashion-MNIST MLP + CNN | 30 configs, seeds 200-202, opt_* diagnostics logged | 90 | `B/analysis_v2/confirm_fm`, `jobs_CONFFM.txt` | `confirm_fm_gen.py`, `step10_confirm_fm.py` |
| Vector field | MNIST, FM, CIFAR-10 | decomposition, spectra, train/test, dataset variants | on the 800-run set | `B/dynamics_v2/vfield`, `.../ntk_coupling` | `latentdyn.vfield.{core,decomp,spectra,spectra_events,traintest,dataset,aggregate,watch}`, `latentdyn.ntk.*` |
| AE grid | MNIST, FM: MLP AE d16 (+d8, d64), linear AE d16 | N {250,1k,4k,full} x wd {0,1e-4} x 3 seeds | 48 per dataset (+ d8/d64 12) | `B/ae_dynamics/runs` (138 dirs), analysis `B/ae_dynamics/analysis` | `scripts/ae/run_ae_grid.py`, `latentdyn.ae.analysis`, `scripts/ae/ae_aggregate.py` |
| AE CIFAR-10 | conv AE d64, linear d64, conv d16 | N x wd | 42 | `B/ae_dynamics/runs`, `analysis/cifar10` | `run_ae_grid.py` (`--encoder_arch convae`) |
| AE dense | MNIST MLP AE | every 2 steps, N {250,1k} x wd x 3 seeds | 12 (+12 null) | `B/ae_dynamics/dense` | `scripts/ae/run_ae_dense.py`, `run_ae_dense_null.py`, `latentdyn.ae.dense_leadlag` |
| AE NTK | MNIST AE | NTK at checkpoints; lr_mult_enc/dec x0.1 control | 21 + 3 | `B/ae_dynamics/ntk` | `run_ae_ntk_ckpt.py`, `run_ae_lrctl.py`, `latentdyn.ae.ntk`, `ae_ntk_analyze.py` |
| AE linear theory | MNIST 10k subset, linear AE | SGD/Adam x default/small init x 3 seeds | 12 | `B/ae_dynamics/linear_theory` | `archive/misc/linear_ae_theory.py` |
| AE topology across regimes | MNIST, FM: MLP + linear AE vs classifier | N {1000,full}, 3 seeds, test probe, ripser H1 | 24 AE runs | `B/ae_dynamics/analysis/topo` | `scripts/ae/ae_topo_cache.py`, `regimes_topology_fig.py` |

## Key findings by study

### 1. Phase 1: classifier latent geometry and topology (MNIST, FM, CIFAR-10; 27 runs each)

Source: `B/report/report.tex` (Results Table, Discussion); corrections in `B/reanalysis_v2/REPORT.md`.

- Raw latent norm inflates roughly 20-40x over training; scale must be removed before geometry is meaningful.
  Fisher ratio, H1 entropy, H1 lifetime sum and PMag are therefore computed on normalized latents.
- Capacity (MNIST, d=2/4/8/16/32): test acc .951/.971/.975/.976/.975; Fisher ratio 5.56/6.49/4.22/3.62/3.55;
  H1 lifetime sum 1.71/6.60/11.53/12.39/12.33 (3 seeds; `report.tex` Table 1).
- Objective (MNIST, d16): probe acc .976 / Fisher 3.62; probe+recon .977 / 2.29; AE probe acc .878 / Fisher 0.60,
  H1 lifetime sum 36.1 (topologically richest). Ordering replicates on FM and CIFAR-10.
- L2 normalization: Fisher 3.62 -> 15.13 on MNIST (4.18x) at equal accuracy. This is MNIST-specific: about 1.04x on
  FM and 0.80x on CIFAR-10 (project memory, phase-3 note; `B/reanalysis_v2/` does not contradict it).
- Re-analysis against external review (`B/reanalysis_v2/REPORT.md`) refuted or weakened several v1 claims:
  - "Accuracy converges before geometry" holds on MNIST only (acc t90 4.3% of training, Fisher 22%); on FM and
    CIFAR-10 Fisher and H1 reach 90% of their change before accuracy (FM Fisher at step 50; CIFAR Fisher t90 21%
    vs acc 41%).
  - The L2 PMag boost (6.54x) is a window artifact: 0.56x under a dataset-shared window; rank correlation of
    original vs shared-window PMag across 27 MNIST runs is -0.93.
  - "Latent geometry decoupled from the generalization gap" is unsupported: v1 train accuracy was a running
    train-mode mean; with eval-mode train accuracy the MNIST gap is 0.7-1.4% (SD 0.19 pts), config-level
    Fisher-gap r = -0.03 [-0.63, +0.85] (n=8 configs, underpowered). CIFAR config-level r(gap, Fisher) = -0.79,
    confounded with capacity (exploratory).
  - Accuracy does not saturate by low capacity on CIFAR-10 (.406/.573/.650/.670/.674 for d=2..32).
  - H1 lifetime sum mirrors Fisher: rho(Fisher, H1 lifetime sum) = -0.91 (MNIST), -0.79 (FM), -0.73 (CIFAR);
    66-73% of H1 lifetime at d16 comes from the bottom-90% of bars. Open positive: MNIST top-5/10 H1 bars are
    independent of Fisher (rho about -0.05).
  - AE online probe accuracy is within 0.9 pts of an offline probe at the end but lags 24 pts over the first 10
    snapshots; use offline probe accuracy for AE timing.

### 2. Slingshot / Adam spikes and phase diagram (MNIST, FM)

Sources: `B/dynamics_v2/` (`phase_summary.csv`, `slingshot_events.csv`, `dense/`, `sgd_lrB/`), `B/dynamics_report/report.tex`
(Sec. 2), `B/analysis_v2/REPORT.md` (v2 step 2: 15/105 runs slingshot, all wd=0).

- No delayed generalization or phase boundary in the (N, wd) grid: everything varies smoothly.
- Earlier "weight decay switches slingshot off" was a detector artifact (absolute loss threshold); events persist
  to wd=1e-4 and repeat 2-3 times per run with a floor-relative detector.
- Per-step logging (12 Adam reruns): every wd=1e-4 run spikes 4-8 times per 40k steps at near-regular 4-11k-step
  intervals; snapshots catch only 27% of spikes. So snapshot-based spike fractions measure sampling chance, not
  whether a run spikes. Two types: wd=0, 1-2 large spikes (peak loss 8-19, Adam second moment below eps); wd>0,
  4-8 mild spikes (peak 0.6-3), no eps involvement. SGD never spikes.
- Linear stability (lr x preconditioned top curvature) does not predict spikes (AUC 0.59).
- NC1 crossover band was a pinv(Sigma_B) artifact (35/45 drops, class-mean condition number about 1e8). Use
  trace ratio or ridge NC1 (`archive/slingshot/nc1_check.py`).
- SGD batch/lr (lrB, 48 runs): better-separated end state at large lr is about 70-75% noise scale lr/B and 25-30%
  step size (log Fisher at t=240: c(log nu) about +0.15/+0.18, b(log lr) about +0.09/+0.06 for N=1000/4000);
  robust to matched epochs and accuracy. 3 seeds, fit-based attribution.
- Label-noise memorization crossover scales with N (about 20-40 epochs); test-acc peak at about 0.3x the crossover.
- Caveat: the 40k/F/eps block conclusions that rest on snapshot event fractions are superseded by the dense block.

### 3. Generalization prediction, pre-registered (analysis_v2)

Sources: `B/analysis_v2/REPORT.md` (updates 1-8), `B/paper_v2/paper.tex` (latest, draft 8), `B/paper_v2/NUMBERS.md`
(source of each number), `archive/paper_v2_figs/paper_stats6.txt`. Test: leave-one-config-out ridge/GBM, increment
of delta R2 from adding latent geometry G to curves + design (B1), targets test accuracy and train/test gap at
cutoffs 10..1000 steps.

- Exploratory (305 MNIST runs, 69 configs): R2(B1) .95-.97 (acc), .69-.76 (gap). Adding G changes ridge R2 by
  -0.010 to +0.006 (acc), -0.019 to +0.042 (gap). Equivalence at post-hoc margins (+-.03 acc, +-.05 gap): 9/10 cells
  plain bootstrap, 5/10 after Holm, 1/10 when the bootstrap refits. Fashion: 0/10. Margins were chosen post hoc.
- Geometry as a design proxy (exploratory, MNIST only): G alone R2 = .84 for accuracy at cutoff 1000; at cutoff 10
  B0 -> B0+G gives +0.71 [+0.53, +0.99] (acc) and +0.54 [+0.30, +0.74] (gap); random projections +0.42/+0.27,
  noise -0.12/+0.06. The lift vanishes once design is known.
- Under extrapolation (leave-one-n-out etc.) full G hurts (LONO acc cutoff 1000 -0.188 vs random projection -0.006);
  compact-4 is least harmful but still -0.187 on LONO gap at 1000.
- Confirmatory MNIST (28 configs, 84 runs, PREREG sha e4fe0682): R2(B1) .77-.82 (acc), .67-.71 (gap). Full G:
  equivalence 0/10, positive 0/10, all ten estimates negative (-0.026 to -0.127), 90% CI wholly below 0 in 5/10.
  Compact-4: equivalent 1/10 after Holm, positive 0/10. GBM gap-at-1000 hint does not replicate (+0.014
  [-0.063, +0.111]).
- Confirmatory FM (30 configs, 90 runs, PREREG sha 65c25a9e, headline one-sided "positive increment"): R2(B1)
  .57-.72 (acc), .01-.22 (gap). Full G positive 0/10 and equivalent 0/10; estimates mixed (8 negative, -0.083
  to -0.009; 2 positive); wholly below zero in 4/10. Compact-4: 0/10 positive after Holm, 3/10 unadjusted
  (Holm p 0.17-0.39). Closest cell to a gain: FM gap at cutoff 1000, low-dim blocks (te_fisher +0.249
  [+0.115, +0.391], Holm p .058); untested at larger n.
- Placebos: the loss is generic to adding about 18 columns. MNIST mean dR2: G -.073, random projections -.015,
  noise -.044, curve-statistics block -.088. FM: G -.037, RP18 -.080, noise -.091, optimizer-statistics block
  OPT18 -.176. G is below random projections in 10/10 MNIST cells (2/10 Holm) but above them in 6/10 FM cells.
- Power: median MDE80 for full G about 0.14 (range 0.10-0.20) in the confirmatory design; gains of 0.03 would be
  missed; about 115 configs would be needed to confirm equivalence (range 42-737, post hoc). Within-configuration
  signal (exploratory +0.05 RMSE points) does not replicate on either confirmatory set.
- Not done: external timestamp of either PREREG hash; confirmatory study on CIFAR-10 (user declined).
- Reading: no detectable forecasting increment of latent geometry over curves and design at configuration level;
  this is a calibrated null with wide intervals, not a proof of zero.

### 4. Vector-field decomposition of latent motion

Sources: `B/dynamics_report/report.tex` (Secs. 3, 5, 8), `B/dynamics_v2/vfield/`.

- Held-out per-sample displacement between snapshots: one global affine map explains 80-97% (shares: 95% steps
  1-10, 97% 10-100, 87% 100-1k, 80% >=1k); class-wise affine maps add 15-28% of the remainder; nonlinear field adds
  about 0 (kNN <= 2.7%). 70-90% of the post-affine residual is not predictable from position, at all lags 1..1000.
- The generator is anisotropic (shear and rotation dominate), which corrects the v1 "isotropic inflation" reading.
- Spectra of class-wise generators decay smoothly with no sign change; eigenvalue-count events match a shuffled
  null; at N <= 4000 late spectra are within the null. No latent phase transitions. Leading eigenvalue scales with
  lr/B for SGD (null grows with noise too).
- Mislabeled points flip from their true class's map to the given label's at 2-15 epochs, before per-class NCM
  memorization (22-28 epochs, constant in epochs).
- Across datasets (`dynamics_report` Table 1): shared-map share 0.87/0.91/0.68 and class-wise share of residual
  0.20/0.28/about 0 (MNIST/FM/CIFAR-10); FM replicates MNIST; CIFAR-10 has no class-wise motion and spectra within
  the null.
- Caveat: R1 conflates nonlinear warping with new input directions entering the latent (row-span issue); late
  R1 below 1 is not itself evidence of nonlinearity. A span-split decomposition is not done.
- Train vs test: same field (train-fit maps transfer to test with R2 loss <= 0.006, visible at N=250 only), but
  train points are more collapsed with a gap that grows as N shrinks (r = -0.92 vs 1/N over 195 runs). The 1/N
  law fails out of sample on 30 FM random configs (R2 = -0.42): say "grows as N shrinks", not a quantitative law.
  Mislabeled points do not drag test points of their given class.

### 5. NTK (tangent-kernel) coupling

Sources: `B/dynamics_v2/ntk_coupling/summary.md`, `B/dynamics_report/report.tex` Sec. 4, Table 1.

- One-step linearization through the tangent kernel is essentially exact: R2 >= 0.997 (MNIST, after step 10),
  >= 0.994 (FM), 0.97-0.99 (CIFAR-10, approximate: augmented batches cannot be replayed).
- Attribution: momentum history dominates the displacement (energy share 0.52-1.0 across phases in
  `summary.md`); batch membership barely matters. Same-class coupling is 1.3-2.6x (MNIST), 1.8-2.8x (FM),
  1.3-1.45x (CIFAR) other-class; this yields the per-class linear structure.
- Train-train coupling exceeds train-test by only about 1% and does not track the loss gap; own gradient is a few
  percent (up to about 10%) of the batch push.
- Conclusion: positions alone are not a closed dynamical system; the residual is sample-specific motion set by x
  via the kernel plus optimizer state. Open: derive the small-N train-point collapse from accumulated own-gradient
  push (untested hypothesis).

### 6. Dense lead-lag (classifier per-step latent runs; AE dense runs)

- Classifier dense latents (`B/dynamics_v2/dense_latent/analysis`): unexplained residual share 0.71-0.93 (median
  0.78) at every lag; energy about lag^2 to 10 steps (coherent), about lag^1 beyond. On FM it is U-shaped (0.53-0.66
  near lag 10). CIFAR-10: class-wise share about 0 at lag 1, 0.18-0.42 by lag 1000; residual about 95% unexplained.
- AE dense (`B/ae_dynamics/dense/analysis/RESULTS.md`, every 2 steps, N 250/1k, 12 runs): no latent-only signature
  has a sharp onset or demonstrated lead over the reconstruction gap. Decoder-tied error/typicality decoupling lags
  the gap onset by 20-60 epochs. One candidate, the train-minus-test affine-residual share (onset epoch 6-14, about
  2-3x before the gap, amplitude 0.02-0.03), failed the null test (`RESULTS_null.md`): same at full N (no gap),
  within the split-half noise floor (sd about 0.01). Net: no latent-only early indicator of AE overfitting.
- Dense runs are bit-identical to the grid at shared steps (`verify.json`).

### 7. Autoencoder dynamics study (2026-10-21 onward)

Sources: `B/ae_dynamics/analysis/{FINDINGS.md,FINDINGS_fashion.md,NN_RATIO_CHECK.md,LDS_COMPARISON.md,topo/REGIMES_TAKEAWAYS.md}`,
`B/ae_dynamics/{ntk,linear_theory,dense}/RESULTS*.md`, `B/dynamics_report/report.tex` Secs. 8-9 (Tables 2-4).

- Overfitting clock (MLP AE, 3 seeds): relative gap opens at 30-50 epochs at every N; test minimum at epoch 120-150;
  then a late rise (MNIST +63/+52/+4.3% at N=250/1k/4k; FM +48/+34/+3.5%; CIFAR-10 +76/+93/+61%). wd 1e-4 cuts
  the rise (N=1k: +20% MNIST, +11% FM, +40% CIFAR) without changing best test error (0.247 vs 0.244 MNIST).
- Linear AE does not overfit beyond the sample-subspace effect (excess about 1/N; rel. gap +18%/+2.8% at
  N=250/1k MNIST) and has no test minimum.
- Linear AE learns PCs in order (rank corr 0.89-0.93 MNIST, 0.98 FM). Mode strength rises with rise time
  proportional to lambda^-1.1 to -1.4 (Adam default init -1.17 +- 0.01). The earlier "lambda^-2" law was an
  artifact of the scale-free R2 readout (times alignment, not strength); report R2 rise only as alignment time.
  MLP staging is weak (0.44-0.64 MNIST).
- Memorization signature: for memorized train points, reconstruction error decouples from latent distance to class
  mean (Spearman +0.03 train vs +0.70 test, N=1k MNIST; FM +0.05 vs +0.67; CIFAR +0.39 vs +0.71, partial).
  Control (`scripts/ae/ae_decoupling_control.py`): linear AE and PCA baselines show no train<test gap
  (|diff| <= 0.03) vs -0.6 to -0.78 for the MLP AE at N=250/1k on MNIST/FM, gone at N=4000, so it is
  memorization-specific there; on CIFAR only about -0.33 persists at N=1k/4k and N=250 is fully explained by the
  baseline. Shafran-style difficulty calibration raises membership AUC by 0.03-0.11 and does not change conclusions.
- Refuted or retracted after controls: (i) "AE train latents more spread than test" (pixel PCA on N train images
  reproduces it, 1.17 vs 1.2 at N=250); (ii) nearest-neighbour-ratio partial r = -0.96/-0.87 (N=250 leverage
  artifact; with N dummies -0.29 FM, -0.07 MNIST); (iii) the dense affine-residual "early lead" (see 6).
- Geometry vs gap: after removing N, no latent summary has a stable partial correlation with the gap across 24
  MLP runs per dataset, signs flip between datasets (|rho| <= 0.2 within (N, wd) groups for velocity-field
  quantities). The N-dummy NN-ratio partial on CIFAR-10 is -0.56 and unresolved.
- AE class structure: input clusters are kept or sharpened (NCM 0.88 MNIST, 0.86 FM, 0.90 CIFAR) and labels only
  weakly (0.82/0.70/0.36); classifier is the reverse (label NCM 0.97/0.86, input-cluster 0.62/0.63).
- Replication on FM and CIFAR-10: replicates the gap vs N/wd, linear no-overfit, PC order, decoupling (FM almost
  digit for digit; CIFAR partial). Differs: CIFAR test minimum is earlier and falls with N (96/57/50 epochs); late
  rise larger; d-dependence of the gap fails on FM; "AE label structure 5x later" is 1.3x on FM.
- NTK mechanism (`B/ae_dynamics/ntk/RESULTS.md`, MNIST, 3 seeds): decoder carries memorization (H1 supported:
  decoder lr x0.1 delays the test minimum from epoch 120 to 469 +- 38 at N=250 and 561 +- 44 at N=1k; encoder x0.1
  only to about 195-204). Own-gradient dominance (H2) is not specific (linear AE equal). Kernel locality (H3) is
  refuted as the discriminator. What discriminates is the train/test coupling asymmetry of the Adam-preconditioned
  loss-gradient kernel: 40x (N=250), 6.5x (N=1k), 1.1x (full N), about 0.9x (linear AE) at epoch 1000;
  correlates with the decoupling (rho = 0.79), correlational only. The late test rise is a small net of large
  cancelling terms and is not explained.
- Vector field AE vs classifier (`LDS_COMPARISON.md`, `dynamics_report` Table 4): early global-affine R1 about 0.97
  in both; late the classifier stays 0.98 at N=1k and drops to 0.58 (MNIST)/0.60 (FM) at full N; the AE drops at
  every N >= 1k (0.81 at N=1k, 0.75/0.76 at full N) and stays 0.94 at N=250. Linear AE late R1 0.39-0.86 (expected,
  row-span). R1 rises with d (0.51/0.75/0.91 for d=8/16/64). Label-wise maps do not beat k-means maps for the AE.
- Topology across regimes (2026-10-26, `topo/REGIMES_TAKEAWAYS.md`, N=1000, test probe, 3 seeds): classifier and
  MLP AE move together for about 1 epoch (H1 entropy, lifetime sum, PMag all fall; MNIST entropy 6.8 -> about 6.1).
  After that the classifier is flat while the MLP AE reverses toward its initial complexity over epochs 1-100
  (MNIST lifetime sum 10.6 -> 30+, PMag 57 -> about 210), before the gap opens (31-36 epochs). Linear AE is flat
  for 1-3 epochs, minima at epochs 16-50. MNIST and FM agree. Train-probe topology not computed.
  Gotcha: the classifier cache's last snapshot (step 20000) is an outlier; exclude it.
- Open: calibrate the reconstruction-error vs difficulty baseline further; causal test of the coupling asymmetry;
  matched CIFAR-10 classifier runs for the AE grid; whether the shared early field is fixed by the input kernel
  alone. Literature search was shallow: frame as "we did not find", not novelty (`B/ae_dynamics/literature/`).

## Where outputs live

All under `B/` unless noted. Never write bulky outputs to `/data` or `/homes`.

| Item | Location |
|---|---|
| Phase-1 sweeps, run dirs | `runs/`, `sweeps/`, `report_assets/` |
| Phase-1 report (LaTeX/PDF) | `report/report.tex` (+ pdf) |
| Re-analysis of phase 1 | `reanalysis_v2/REPORT.md` and CSV/PNG; code `archive/reanalysis/` |
| v2 runs and launch scripts | `runs_v2/runs/`, `runs_v2/*.sh`, `jobs_*.txt`, `runs_v2/logs/` |
| v2 prediction analysis | `analysis_v2/REPORT.md` (cumulative, 8 updates), `confirm/`, `confirm_fm/` (PREREG + results), `ext*/`; plan in `archive/analysis_v2/ANALYSIS_PLAN.md` |
| Slingshot, vfield, NTK, dense | `dynamics_v2/{phase_*.csv,slingshot_events.csv,dense,dense_latent,vfield,ntk_coupling,sgd_lrB,summary_figs}`; opt-state dumps in `/data2/lbastian/latent-dynamics/opt_dense/` (cap about 40 GB) |
| AE study | `ae_dynamics/{runs,analysis,dense,ntk,linear_theory,literature}`; `analysis/FINDINGS*.md`, `summary_by_config.md`, `topo/`, `lds/` |
| Latent caches | `analysis_v2/cache`, `dynamics_v2/phase_cache`, `reanalysis_v2/cache`, `ae_dynamics/analysis/_cache` |

Current documents:
- Paper (pre-registered null): `paper_v2/paper.tex` (draft 8; `paper_draft7.tex` is the previous draft), numbers in
  `paper_v2/NUMBERS.md` (its header says draft 6 but it carries Draft 7 and 8 sections); earlier paper in `paper/`;
  literature positioning in `paper/lit_v2/POSITIONING.md`.
- Empirical report on dynamics, vfield, NTK, train/test and AE (current for sections 4-7): `dynamics_report/report.tex`
  (+ pdf, figures in `dynamics_report/figures/`) and the one-pager `dynamics_report/onepage.tex`. Compile with
  `/data2/lbastian/conda/envs/latent-dyn/bin/tectonic -X compile report.tex`.
- Writing style: `docs/writing/WRITING_GUIDE.md` (no "first" or priority claims; few dashes).

## Discrepancies and caveats found while compiling

- `paper_v2/NUMBERS.md` is titled "draft 6" but also holds Draft 7/8 notes; `paper.tex` (22:26 on 16 Oct) is one
  minute newer than `paper_draft7.tex` and equals draft 8. Use `paper.tex`.
- Power MDE for full G: 0.142 median over ten cells in draft 6 notes; Draft 7 notes say 0.1415 is the median over
  the 8 of 10 cells where 80% power is reached, and about 0.17 counting unreached cells as above 0.20.
- Own-gradient share of the batch push: 1-2% in the earlier notes, "a few percent, up to about 10%" in the current
  report (FM kernel replication); use the latter.
- Dense lag share: "flat" on MNIST, U-shaped on FM (0.53-0.66 near lag 10); the earlier "non-diffusive" wording was
  corrected to "never averages out".
- The 1/N train/test collapse law was stated as a law early on; it fails out of sample (R2 = -0.42 on FM random
  configs).
- `dynamics_report` says about 800 runs; `runs_v2/runs` holds about 926 directories (includes unfinished or held
  jobs from `jobs_E.HOLD.txt`; check `results/runs.csv`).
- `ae_dynamics/analysis/FINDINGS_cifar10.md` was never written (a harness refusal per project notes); the CIFAR AE
  results live in `dynamics_report/report.tex` Table 2 and `analysis/cifar10/`.
- Classifier topology cache has an outlier last snapshot (step 20000); cause not investigated.
- Phase-1 v1 claims dropped after review: accuracy-first on all datasets, saturation on all datasets, the 6x PMag
  boost, decoupling from the gap, "fine temporal resolution".


## Corrections (2026-10-26)

The original inventory and findings above are retained as historical text. The
notebook back-fill records later evidence and distinguishes report claims from
new metadata/table checks.

- **AE NTK count:** 21 checkpoint runs **plus 12** learning-rate controls (33), rather than 21+3; all 33 config paths exist and the registry marks them complete. See [AE NTK](../notebook/2026-10-22_095841_ae-ntk.md). This is not a fresh audit of every final checkpoint.
- **Train-probe topology exists:** the directory now contains 72 test and 42 train NPZs (partial train coverage), rather than the earlier test-only statement. The newer rebound analysis reuses 72 AE and 30 classifier runs, expanding the original 24-AE regime view. See [topology rebound](../notebook/2026-10-26_184507_ae-topology-rebound.md). The step-20000 classifier outlier remains excluded and unexplained.
- **Rebound analysis added:** REBOUND_VERDICT/TAKEAWAYS and rebound_numbers.json (22:50–22:54 UTC, newer than this digest) separate optimizer-time rebound from data-pass overfitting. H1 minimum is around steps 7–8, half recovery 56–70; pooled MLP rebound-gap rho=-0.08, and weight decay reduces the gap without removing rebound. This does not support interpreting rebound as overfitting. The e90 late tail is not step-clock invariant; train/test and dataset/N caveats remain. See [the entry and preserved sources](../notebook/2026-10-26_184507_ae-topology-rebound.md); scientific claims were not recomputed from raw arrays.
- **AE inventory count:** 138 grid runs already includes dimension ablations (48 MNIST, 48 FM, 42 CIFAR); the additional two smoke runs are separate. Avoid adding d8/d64 runs to 138 again. See [grid and controls](../notebook/2026-10-21_153142_ae-grid-memorization.md).
- **Scope of verified counts:** 926 is the registry count for all v2 training runs, not the number of processed vector-field runs. The preserved core table verifies 105 unique MNIST runs/210 probe rows; replication processed counts remain unaudited. The lrB registry has 48 runs; the digest's +24 matched runs have not been independently identified as a separate run set. See [vector field](../notebook/2026-10-16_160301_vector-field.md) and [lr/batch](../notebook/2026-10-16_101315_v2-lr-batch.md).
