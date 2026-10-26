# Analysis plan v2 (written before any outcome was inspected)

Data: runs_v2/runs, only runs with model_final.pt. MNIST block 105 runs = 35 configs x 3 seeds
(25 "phase" configs: n in {250,1000,4000,16000,full} x wd in {0,1e-4,1e-3,1e-2,3e-2}, ln=0;
10 "gap" configs: n x ln in {0.2,0.4}, wd=0). CIFAR-10 ResNet-18 block: descriptive only.
Unit of inference = CONFIG (seeds are replicates, never independent units). Binding lessons from the v1 audit:
fixed-grid Mag(t) on mean-pairwise-normalised latents, fixed-window PMag, H1 total AND top-5/top-10,
offline linear-probe accuracy for "accuracy of the representation", config-level bootstrap CIs.

## Per-snapshot metrics (both probe sets: test = latents/, train = latents_train/)
fisher (true labels; plus noisy labels for train probe with ln>0), raw mean L2 norm, offline 5-fold linear-probe acc
(true labels; plus noisy labels on train probe), effective rank = participation ratio of centred-latent covariance
eigenvalues (+ entropy effective rank), H0 total lifetime (finite bars), H1 total, H1 top-5, top-10 lifetime sums,
H1 persistence entropy, Mag(t) on t = geomspace(0.1,100,19) of mean-pairwise-normalised latents, Mag(t=1),
PMag over the FIXED window t in [0.5,10] (12 linear points, trapezoid, mean-distance units).
Cheap metrics on all ~120 snapshots; topology + magnitude on a subset: all steps <= 10 plus every second snapshot
after, plus the final one (~66 snapshots). Full 1000-point probe sets (no topology subsampling) unless timing forces it.
CIFAR D=512: metrics on the normalised cloud (ripser on the distance matrix).

## Slingshot handling
onset from dynamics_v2/phase_summary.csv (slingshot_onset_step; NaN = none), else own detector (train loss jumps >=3x after
<1e-4 for >=3 snapshots). "pre-slingshot" snapshot = last snapshot with step < onset (final if none).
Outcomes reported at fixed steps {1000,5000,15000,final} and pre-slingshot.

## Step 2 (descriptive)
(a) Delta_m(t) = m_train - m_test per config (mean over seeds); onset-timing of Delta_m vs accuracy gap measured as
half-max time (log-step interpolation of normalised progress p(t) = (x_t-x_0)/(x_T'-x_0), T' = last pre-slingshot).
(b) heatmaps over n x wd (and n x ln) at fixed steps of test acc, gap, Delta fisher, Delta probe acc, Delta H1 top-5, Delta eff-rank.
(c) baseline-normalised convergence: half-max time t50 of p(t) for test acc, train acc vs geometry metrics; per config
sign of log10 t50(metric)-log10 t50(test acc) (geometry-first if negative), summarised across configs with bootstrap CI.
(d) config-level Spearman correlation matrix (test probe, step 5000), partial correlation with Fisher; bootstrap CI over configs.

## Step 3 (main test: early prediction)
Rows = runs (105), groups = configs (35). Targets: gap (train_acc - test_acc, eval-mode full-train accuracy vs noisy labels
if ln>0) and test acc at (i) step 15000 snapshot, (ii) last pre-slingshot snapshot. Cutoffs {10,30,100,300,1000}: for each
run, the nearest topology-subset snapshot with step <= cutoff supplies ALL features.
Feature sets:
 B0 = baseline curves: log train loss, log test loss, train acc, test acc, gap, log raw norm (test probe).
 B1 = B0 + design vars: log10 n (full=60000), log10(wd+1e-4), label noise.
 G  = geometry: for each probe set {fisher(log), probe acc, eff rank(log), H1 top-5 (log), Mag(t=1)(log), PMag(log)} and
      train-minus-test difference of each (18 features).
 Compared: B0, B1, B0+G, B1+G (primary honest test: B1 vs B1+G), and G alone, G+design.
Models: ridge (standardise inside fold; alpha by nested GroupKFold(5) over training configs, alphas logspace(-3,3,13));
check: GradientBoostingRegressor (depth 2, 150 trees, lr 0.05, subsample 0.8).
CV: leave-one-CONFIG-out. Metrics: pooled out-of-fold R^2 and Spearman. Increment = R2(B+G) - R2(B): 95% CI by paired
bootstrap over configs (2000 resamples of the 35 configs from the OOF predictions); permutation test: permute the G block
across configs (all seeds of a config move together; 500 perms for ridge, 0 for GBM) and refit, p = fraction of perms with
increment >= observed. Report R2(cutoff) curves for all feature sets. Sensitivity: exclude runs whose slingshot onset <= 15000.
Verdict rule: geometry "adds information" only if the B1 -> B1+G increment CI excludes 0 AND permutation p < 0.05, in ridge,
consistent in sign for GBM.
CIFAR: descriptive only (Delta curves; values at 5%/25%/50%/100% of training steps), no CV.

## Amendment 2026-10-15 (before fitting; from coordinator context, not from my own outcome inspection)
- Slingshots only at wd=0 (onset ~16.6k n250, ~20k censored n1000, none N>=4000 except ln=0.4 occasional). Use phase_summary
  slingshot_onset_step. Sensitivity (exclude onset<=15000) retained.
- Label-noise runs are two-phase (test-acc peak, then memorisation; crossover scales with N, ~40 steps at n250 up to 15-19k at full data).
  For full-data ln>0 configs the step-15000 target sits mid-transition: I report targets at 15000 AND at pre-slingshot/final, and list
  which configs are mid-transition (crossover within +-30% of 15000 steps). Early cutoffs (<=1000) are pre-memorisation for large N: kept.
- Added EXPLORATORY descriptive targets for ln>0 runs: peak test acc, peak step, memorisation-crossover step (first step with
  gap = train_acc - test_acc > 0 after the initial phase, i.e. first upward zero-crossing); prediction of peak test acc from early
  features is run with the same machinery, labelled exploratory.

## Amendment (extension) 2026-10-15 (written BEFORE any extended-set outcome was inspected; audit of the original result ran in parallel)
Code: ext_audit*.py (audit), step3_predict_ext.py / step2_ext.py (extension); common2.py now has ANALYSIS_RUNSET={orig,ext,ext_adam} and ANALYSIS_OUT=<name>
(-> analysis_v2/ext/<name>/); per-run caches are shared (cache/), added to only. Original step2_*/step3_results.csv untouched.
- New blocks: C = sl40k_n{250,1000,4000}_wd{0,1e-6,3e-6,1e-5,3e-5,1e-4} (Adam, 40000 steps, 140 snaps, seeds 0-4); D = sgd40k_n{250,1000,4000}_wd{0,1e-4}
  (SGD mom 0.9 lr 0.05, 40000 steps, seeds 0-4); E2 = phase_n{...}_wd{3e-4,3e-3} (Adam, 20000 steps, seeds 0-7). Only runs with model_final.pt.
  Run-set blocks A/B = original 105 runs. config_id gets suffix "|optimizer|train_steps" only when (optimizer,steps) != (adam,20000), so C/D never merge
  with A/B configs of the same n,wd; original ids unchanged.
- Sets: (i) orig 35 configs (must reproduce), (ii) ext = A+B+C+D+E2, (iii) ext_adam = ext without D.
- Design variables: orig keeps the 3 (log10 n, log10(wd+1e-4), label noise). Ext sets use DES_ext = those 3 + is_sgd + log10(train_steps)
  (constant columns are inert after standardisation, so ext_adam effectively has log10 steps only).
- Cutoffs {10,30,100,300,1000}; features as in the plan (nearest topo snapshot <= cutoff). Feature sets, models (ridge+nested group CV, GBM), LOCO, config
  bootstrap (2000) and G-block permutation (300 ridge perms) unchanged. Rows = runs, groups = configs (E2 configs carry 8 seeds, C/D 5, A/B 3, so pooled
  R^2 weights configs by their seed count; sensitivity: restrict every config to seeds 0-2).
- Targets: gap & test acc at (a) step-15000 snapshot [point], (b) 15k WINDOW target = median over snapshots with 12000<=step<=18000, (c) final point (last
  snapshot; = 40k for C/D, 20k for A/B/E2), (d) FINAL WINDOW = median over snapshots with step >= 0.9*train_steps. Window targets are the robustness variant for
  loss-spike (slingshot) contamination of point targets. For 40k-step runs (C,D) the (c)/(d) targets are additionally reported on the 40k-only subset
  (tag final40k; ext_adam: C only), since there "final" is 40k.
- Slingshot-dependent items (pre-slingshot targets gap_pre/acc_pre; sensitivity excluding onset<=15000): the existing phase_summary.csv slingshot columns are
  biased (absolute <1e-4 loss floor; misses spikes at wd>=3e-5; own-detector fallback for unlisted runs has the same flaw). These items are PROVISIONAL and will
  be computed only once phase_summary.csv has a slingshot_detector_mode column (floor-relative detector); until then they are not reported for the extended sets.
  The window targets are the primary spike-robust analysis.
- Step 2 on ext sets: per-config own step grid (40k runs have 140 snaps), outputs (a) delta timing, (c) t50 comparison, (d) redundancy at step 5000, config summary;
  heatmap grids/figures only for the original set. Slingshot bookkeeping columns in step 2 are provisional for the same reason.
- Audit-only analyses (not part of the verdict): compact geometry sets, alternative alpha grids/GBM settings, delta-target (target - value at cutoff),
  positive/negative controls.
- (14:50) phase_summary.csv was regenerated with slingshot_detector_mode='floor' for the original 105 runs only. Pre-slingshot targets / onset<=15k sensitivity are
  therefore enabled for set (i) (all runs covered) and stay disabled for sets (ii)/(iii) until C/D/E2 runs appear in the file with a detector mode (rule: has_detector_mode
  requires every run in the set to be covered). Note that set (i) 'pre' now uses the new floor-relative onsets (earlier 'pre' results in REPORT.md used the old column).
- (15:05, still before extended outcomes) Slingshot onsets now come ONLY from the floor-relative (pd-5) files: dynamics_v2/{phase_summary,sl40k/phase_summary,sgd40k/phase_summary}.csv
  (old absolute-threshold fallback removed; runs not in these files, currently E2, are 'uncovered' and get no onset). Pre-slingshot target in two variants:
  gap_pre/acc_pre = last snapshot before the ANY-event onset (slingshot_onset_step); gap_preN/acc_preN = before the first type-N (norm-growth) onset
  (slingshot_onset_step_N). Onset<=15000 exclusion uses the any-event onset. These items are run on the covered subset (tag pre_covered[_subset], sens_excl_slingshot_le15k[_subset]).
  Caveat for the report: late snapshots are ~7% apart, so short spikes can be missed; type-F (floor-driven) events at wd>0 also count as any-event.
- Added set (iv): SGD-only (block D, no slingshot events observed) as a clean replication (ANALYSIS_RUNSET=sgd). In set (ii) optimizer stays a design variable (is_sgd).
- Adam 40k-final targets (gap_fin/acc_fin/finw, final40k tag) are flagged slingshot-affected (repeated events at wd<=1e-4); 15k and window-median targets are preferred.

## Amendment 3 (reviewer analyses) -- 2026-10-15, written BEFORE running any of the analyses below
Context: the set (ii)/(iii) step-3 results in REPORT.md were already known when writing this (so "primary" cells are not blind), but none of the
analyses below (equivalence, extrapolation splits, residual targets, B2, power simulation, timing sensitivity, multi-step redundancy, PMag windows,
Fashion replication) has been run or inspected. New code = step4_*.py (shared helpers: step4_lib.py); outputs = analysis_v2/ext2/{mnist,fashion}/.
Existing outputs and runs_v2/ are never modified. CPU only, <=8 BLAS threads/process, <=4 processes. Unit of inference = config; all bootstraps resample configs
(seeds of a config move together), 2000 resamples (500 inside the power simulation), seed 0.
Set (ii) = ANALYSIS_RUNSET=ext (305 MNIST runs / 69 configs), design DES_ext (log10 n, log10(wd+1e-4), label noise, is_sgd, log10 steps), features and cutoffs
{10,30,100,300,1000} exactly as step3_predict_ext.py (nearest topo snapshot <= cutoff). Models: ridge (nested group-CV alpha, as before) and GBM (depth 2, 150 trees)
wherever cheap (analyses 1, 2, 4; ridge only in 3 and 5). Primary targets: acc_15kw and gap_15kw (12k-18k window medians); the point targets acc_15k, gap_15k are
the "point" variants in analysis 1.

### A1. Equivalence (TOST-style) on increments
Cells: target in {acc_15k, gap_15k, acc_15kw, gap_15kw} x cutoff (5) x model (ridge, GBM) x comparison in {B1 -> B1+G, B2 -> B2+G (see A4)}.
dR2 = R2_OOF(B+G) - R2_OOF(B) from leave-one-config-out predictions; paired config bootstrap (2000); 90% CI = [5th, 95th] percentile (TOST at alpha 0.05).
Pre-stated margins: delta = 0.03 R2 units for accuracy targets, 0.05 for gap targets. Classification per cell:
 positive      : 90% CI lower bound > 0 (one-sided 5%); (flag "also within margin" if the CI is also inside +-delta)
 equivalent    : not positive and 90% CI subset of (-delta, +delta)
 inconclusive  : otherwise (includes CIs reaching below -delta, i.e. possible harm, flagged "possible harm")
Also reported: smallest symmetric margin m* = max(|lo90|,|hi90|) that would yield equivalence, and the 95% CI. Verdict statements are made per cell, never pooled.

### A2. Extrapolation splits (targets acc_15kw, gap_15kw; all cutoffs; ridge + GBM)
 (a) leave-one-n-out: n in {250,1000,4000,16000,full}; train on configs with the other four n, test on held-out n (pooled OOF over the five folds; per-fold R2 also given).
     Design variable log10 n is extrapolated linearly for ridge. Alpha chosen by group-CV over training configs only.
 (b) leave-one-wd-level-out: every wd level with >=4 configs in the set; pooled OOF over those levels and per level. (wd=0 contains all label-noise/gap configs and the
     slingshots, so it is a hard extrapolation; reported but flagged.)
 (c) leave-one-optimizer-out: train on all Adam configs (63), test on the 6 SGD configs (30 runs). is_sgd is constant in training and inert; log10 steps kept.
 Per split: R2(B1), R2(B1+G), dR2 with 95% config-bootstrap CI (+ 90% CI and equivalence label with A1 margins), and R2(G alone). Bootstrap resamples held-out configs.
 R2 is computed about the held-out set's own mean (pooled OOF for pooled rows), so R2 can be negative.

### A3. Ceiling-free targets (ridge, leave-one-config-out; cutoffs all; targets acc_15kw, gap_15kw)
 (a) within-config seed residual: y_c = y - config mean(y); X_c = X - config mean(X) (feature means over the seeds of the same config); design columns vanish (constant
     within config), so base = B0 (centred) vs B0+G (centred). All configs have >=3 seeds. Features are centred with the config's own seed features only (no target info).
     R2 on the centred target; dR2 + 95% config-bootstrap CI; also reported: mean within-config SD of the target, Spearman of OOF vs truth. Equivalence margin as A1.
 (b) design-residual target: inside each training fold, regress y on DES by OLS (variant "lin") and on DES plus squares and pairwise products (variant "quad");
     the residual r = y - f_design(DES) is the new target (for the held-out configs r is formed with the training-fold f). Compare B0 vs B0+G for predicting r
     (ridge, nested alpha). R2 is relative to the mean of r on the held-out configs (pooled OOF); 95% config-bootstrap CI for dR2.

### A4. Stronger curve baseline B2 = B1 + learning-curve extrapolation
For each run, using ALL snapshots with step <= cutoff (not only topo), fit the test-accuracy curve and the gap curve with three families (t' = step+1):
 pow3: y = c + a * t'^(-alpha), alpha on a 40-point log grid in [0.03, 5], (c,a) by linear least squares per alpha (variable projection), best SSE;
 exp : y = c + a * exp(-lambda t'), lambda on a 40-point log grid in [1e-4, 3] (per step) , same;
 loglin: y = a + b * log t'.
Extrapolate to step 15000 (centre of the 12k-18k window), clip to [0,1] (acc) and [-0.3,1] (gap). Fallbacks: <4 distinct points -> persistence (last observed value) for pow3/exp,
loglin with >=3 points else persistence; non-finite or failed fit -> persistence. B2 adds the 6 extrapolated values (3 families x {acc, gap}) to B1.
Report R2(B2), dR2(B1->B2), dR2(B2->B2+G) with CIs and the A1 classification, ridge + GBM.

### A5. Power by simulation (full set (ii), ridge, B1 base)
Targets acc_15kw and gap_15kw, cutoffs 100 and 1000. e = y - OOF_B1(y) (LOCO ridge). Synthetic feature s = rho * z(e) + sqrt(1-rho^2) * z(n), where n = a real G-like noise column
(a randomly chosen real G column, config-permuted across configs, then orthogonalised against e) so that corr(s, e) = rho exactly in sample, rho in {0.2,0.3,0.5,0.7}
(plus rho=0 as a false-positive calibration). 60 replicates per cell (new noise column + permutation each). Pipeline: B1 vs B1 + full G(18) + s, and B1 vs B1 + compact G (8: te+delta of Fisher,
eff rank, Mag(t=1), H1 top-5) + s; detection = 95% config-bootstrap CI (500 resamples) lower bound > 0. Report detection probability, mean dR2 per cell, and the minimum
detectable dR2 = mean dR2 at the smallest rho with detection probability >= 0.8 (linear interpolation between rho levels of detection probability vs mean dR2; "not reached" if never).
Note: the synthetic feature is built from out-of-fold residuals, which already contain out-of-fold B1 information only; it is a deliberately optimistic signal.

### A6. Timing sensitivity (step-2 train-test divergence timing, set (ii))
Recompute log10 t50(Delta_m) - log10 t50(gap) (and Delta_m-vs-test-acc half-times) with progress thresholds q in {0.25, 0.5, 0.75} instead of only 0.5, and (b) excluding every run with a finite
slingshot onset (floor-relative pd-5 files; runs not covered by these files (block E2) are treated as unknown and dropped from the exclusion variant, so that variant is also reported on covered runs without
exclusion for comparability). Config-level medians + bootstrap CI (4000). For every (metric, q) record configs with undefined timing and the reason (flat/no change |dx|<1e-9; transient: |final
change| < 0.5 x max excursion; level never reached; <3 finite points); compare design (n, wd, label noise, optimizer) of defined vs undefined configs (counts, median log10 n, Mann-Whitney p).

### A7. Redundancy at multiple steps
Spearman rho (config level, test probe, config mean over seeds, topo metrics from nearest topo snapshot <= step) of Fisher with H1 total, H1 top-5, Mag(t=1), PMag at steps 300, 1000, 5000, 15000;
bootstrap over configs (2000) 95% CI. Reported for set (ii) (69) and set (i) (35).

### A8. PMag window rho across datasets
From reanalysis_v2/pmag_windows_final.csv: Spearman of original (pmag_csv) vs dataset-shared window (column B) final PMag over the 27 D=16 runs per dataset (MNIST, Fashion, CIFAR-10), with n and
a bootstrap CI over configs (objective/norm; 9 configs per dataset). Also the other variants (Wd, U, mag_t1) for completeness. No new computation of PMag.

### A9. Fashion-MNIST replication (blocks A+B: 105 runs, 35 configs, Adam 20k)
Metrics: compute_metrics.py --runs <fashion run ids> after verifying that probe sets / dataset handling are right (shapes 1000x16, labels 0-9, noisy_labels present iff ln>0, same snapshot grid as MNIST,
train probe subset drawn from the training subset). Analysis on Fashion alone, design = the 3 original variables, no slingshot-dependent items (no floor-relative onset files for Fashion):
 - main step-3 test: B1 vs B1+G, ridge + GBM, all 5 cutoffs, targets acc_15kw, gap_15kw (plus point acc_15k, gap_15k), LOCO, 2000-bootstrap, 300-permutation (ridge) as in step 3;
 - A1 equivalence (same margins), A2(a) leave-one-n-out (5 folds x 7 configs), A3(a),(b), and A4 (B2) as an extra, all with the same definitions as for MNIST.
Pre-stated replication criterion: the MNIST conclusion "no increment beyond B1 for accuracy; at most weak/unresolved for gap" replicates if for acc the dR2 CI includes 0 or is equivalent, and for gap the
dR2 CI lower bound <= 0 in ridge. A positive CI lower bound (>0) at any cutoff would be reported as a (non-)replication, not explained away. Fashion queue completion rule: log ends with "QUEUE DONE" and 105 OK lines.

## Amendment 4 (review 2 analyses) -- 2026-10-15 (late), written AFTER all previous results (Amendments 1-3, REPORT UPDATE 2, review2.md) were seen
Honest status: everything below responds to review2.md and is specified before it is run, but the primary cells, margins and the geometry-hurts-under-extrapolation
finding were already known; nothing here is blind. Code = step5_*.py (helpers step5_lib.py, imports step4_lib); outputs = analysis_v2/ext3/{mnist,fashion}/.
Existing outputs never modified. CPU only, <=4 BLAS threads/process, <=4 processes. Datasets: MNIST set (ii) (305 runs/69 configs; blocks A,B,C,D,E2) and Fashion
(105 runs/35 configs; blocks A = phase configs, B = gap/label-noise configs; block label derived from run_id). Targets acc_15kw, gap_15kw (primary), acc_15k/gap_15k (point,
only in analyses 1-3); cutoffs 10/30/100/300/1000. Models/definitions of B0, B1, G, LOCO, ridge nested alpha, GBM as in Amendment 3. Config bootstraps use the same
resampling unit (configs, seeds together) but are implemented from per-config sufficient statistics (so numbers differ from the earlier 2000-rep/seed-0 CIs only by Monte Carlo error).

### A4.1 Margin sensitivity
Cells = dataset x model (ridge, GBM) x target (4) x cutoff (5), comparison B1 -> B1+G (LOCO OOF). Bootstrap 10000 config resamples of dR2. TOST p-value at margin m =
max(P*(dR2* <= -m), P*(dR2* >= m)) with (count+1)/(nb+1); computed on m = 0..0.15 step 0.005. m* = smallest m with TOST p <= 0.05 = max(|q05|,|q95|) (90% CI; exact value, plus the
grid curve). Also the "bootstrap mass in (-m,m)". Substantive reference: with y variance V (pooled), RMSE_B1 = sqrt((1-R2_B1) V), the dR2 equivalent to changing RMSE by 0.5 accuracy
points (0.005) or 0.5 gap points (0.005): dR2_improve = (RMSE^2 - (RMSE-0.005)^2)/V, dR2_degrade = (RMSE^2 - (RMSE+0.005)^2)/V (negative); reported per cell with RMSE_B1 in points.
### A4.2 Design-aware uncertainty (ridge, B1 -> B1+G; GBM only in (a))
(a) Block-stratified config bootstrap (configs resampled within block, block sizes fixed), 10000 reps, 95%/90% CI, next to the unstratified CI.
(b) Full-pipeline refit bootstrap: resample configs with replacement; refit everything on the resample (LOCO where all copies of the held-out config are removed from training,
nested group-CV alpha selection on training copies grouped by original config, for both B1 and B1+G); dR2 computed with multiplicity weights on the held-out OOF predictions.
Percentile CI. 300 reps per cell (MNIST: both targets x all 5 cutoffs; Fashion likewise) -- count reported in the output; more if time allows.
(c) Block-permutation test of G: G rows (config-wise, seeds mapped as in permute_G) permuted among configs within the same block (300 perms ridge; 1-sided p = (#perm dR2 >= obs +1)/(n+1)),
reported next to the unrestricted permutation p. (Caveat: shuffled-G null is weak, see audit.)
### A4.3 Multiplicity
Distinct targets: accuracy and gap (window and point variants of the same quantity are treated as one). Equivalence summary per distinct (target, cutoff) = window variant (point listed
beside it). Holm-adjusted TOST over the 10 primary ridge cells (acc_15kw, gap_15kw x 5 cutoffs) per dataset using the pre-stated margins (0.03/0.05) and the A4.1 TOST p-values; a cell
remains "equivalent" if its Holm-adjusted p <= 0.05.
### A4.4 Learner and feature-set sensitivity, main LOCO test and extrapolation splits (leave-one-n-out, leave-one-wd-out [MNIST only, as before], Adam->SGD [MNIST])
G variants: full (18); compact4 = test-probe {fisher, eff_rank_pr, mag_t1, h1_top5}; compact8 = compact4 + train-test deltas; pca2, pca4 = first 2/4 PCs of the standardised full G block
(fit on training fold only). Learners: ridge (nested alpha), kernel ridge (RBF, inner 5-fold group-CV over gamma in {0.01,0.03,0.1,0.3,1}/n_features and alpha in logspace(-3,1,5),
standardised X, centred y), MLP (sklearn MLPRegressor, 1 hidden layer of 16, adam, early_stopping, standardised X and y, 5 restarts with different seeds, predictions averaged). Each learner
is compared with ITSELF: dR2 = R2(learner on B1+Gvariant) - R2(learner on B1). Targets acc_15kw, gap_15kw, all cutoffs. Fashion: LOCO + leave-one-n-out only (no wd/optimizer folds).
Question: does "geometry hurts under extrapolation at cutoffs >= 100" survive with compact/shrunk geometry (criterion: pooled 95% CI upper bound < 0)?
Extreme-negative-R2 diagnosis: per fold the held-out variance, n, and, besides R2 about the held-out mean (existing), (i) R2 about the training-fold mean, (ii) R2 about the pooled grand
mean of all runs, (iii) RMSE ratio RMSE(B1+G)/RMSE(B1) (bootstrap CI over held-out configs) and RMSE in accuracy/gap points. dR2 variants (i),(ii) get 95% config-bootstrap CIs.
### A4.5 Residual targets with flexible design models (ridge; B0 -> B0+G on the residual target; LOCO; nested alpha; cutoffs all; targets acc_15kw, gap_15kw)
Design-residual r = y - f(DES) with f fitted in the training fold: (i) linear OLS, (ii) quadratic (squares + products, ridge 1e-2) as in step4_resid, (iii) GBM on DES (depth 2, 150 trees;
training residuals are cross-fitted by an inner 5-fold group split to avoid in-sample shrinkage, test residuals use the full-training-fold fit), (iv) saturated config-mean model = the
within-config residual (target and features centred on the held-out config's own seeds, as step4_resid 3a). One table with centred and design-residual rows. dR2 with 95% config bootstrap.
### A4.6 Power simulation v2 (MNIST set (ii), ridge, B1 base, cutoffs 100 and 1000, both primary targets)
Replicates: y_sim = P_B1(y) + e_perm + b*s, where P_B1 is the in-sample OLS projection of y on B1, e_perm the residual permuted across configs (seed-matched, permute_G style), and s = a real
geometry feature at the cutoff (te_eff_rank_pr and, second, te_fisher) regressed on B1 and residualised (s orthogonal to B1 incl. intercept, unit variance). b is solved per replicate so the
true in-sample incremental R2 of s over B1 equals r in {0 (calibration), 0.02, 0.05, 0.10, 0.15}. Real pipeline: LOCO ridge B1 vs B1 + full G and vs B1 + compact8 G (G contains the raw feature);
detection = 95% prediction-bootstrap CI (500) lower bound > 0. 200 replicates per cell. The earlier residual-shaped variant (rho in {0,.2,.3,.5,.7}) is rerun at 200 replicates. Report
detection curves, mean dR2 (vs true r: ridge shrinkage / overfit cost) and the minimal r with >= 80% detection.
### A4.7 Learning-curve baselines on gap (and acc for contrast), B1 -> B1+X, ridge and GBM, then X+G
Per run, from ALL snapshots with step <= cutoff: iso = increasing isotonic regression of gap vs snapshot order, value at last snapshot (monotone-smoothed current gap); med3 = median of last 3
snapshots; slope = OLS slope of gap vs log10(step+1) over the last 5 snapshots (0 if < 3 points); slope_dec = same over snapshots with step >= cutoff/10. Sets: B1+iso, B1+slope(+slope_dec),
B1+iso+slope(+slope_dec), each also +G. dR2 and 95% CI vs B1 and vs the augmented base. Targets gap_15kw and acc_15kw.

## Amendment 5 (review 3 analyses, post hoc) -- 2026-10-15 (night), written AFTER REPORT UPDATE 3 and review3.md were seen; before the step6 outputs were inspected
Honest status: all primary cells, margins, the "full G hurts under extrapolation / compact neutral" finding and the Fashion GBM gap-1000 signal (+.254 [.034,.546]) were known. Nothing is blind.
Code = step6_*.py (helpers step6_lib.py -> step5_lib -> step4_lib); outputs = analysis_v2/ext4/{mnist,fashion}/. CPU only, 2 BLAS threads/process, <=4 processes, loadavg watched. Existing outputs never modified.
Datasets/targets/cutoffs/LOCO/config bootstrap exactly as Amendment 4 (MNIST set (ii) 305 runs/69 configs; Fashion 105 runs/35 configs; acc_15kw, gap_15kw; cutoffs 10/30/100/300/1000; ridge nested-alpha and GBM d2/150).
Bootstraps resample configs (seeds together) from per-config sufficient statistics (10000 reps unless noted; 4000 for per-draw placebo and per-fold rows).
### A5.A Placebo blocks (step6_feats.py, step6_oof.py, step6_analyze.py)
Inspection of the cache npz keys: the per-run files hold only loss/acc/epoch/lr scalars and latent metrics, i.e. no weight/gradient norms. Candidate (i) therefore reduces to the train-probe raw norm (+ train-minus-test
norm difference), a 2-feature block ("Pnorm"), reported but not size-matched. Main placebos (matched sizes: 18 = full G, 8 = compact8, 4 = compact4):
 (ii) RP block: for each draw (5 draws ridge, first 3 for GBM; fixed seed) six random unit directions w_j in R^16 (one per statistic type); the cutoff snapshot's probe latents (same snapshot as G: nearest topo snapshot <= cutoff;
      test and train probe) are projected, z_j = x w_j, and summarised by 6 statistics: log std, skewness, excess kurtosis, mean, log mean-absolute-deviation about the median, log IQR. Block = test (6), train (6), train-minus-test (6) = 18;
      compact8 = first four stats of test + first four deltas; compact4 = first four test stats. Information available at the cutoff only; no labels used.
 (iii) NZ block: iid N(0,1) columns, same sizes, same number of draws.
Comparisons (all LOCO OOF, dR2 = R2(comp) - R2(base) about the evaluated set's mean; 95% config bootstrap; placebo rows report the mean over draws of dR2 and of the CI bounds, the min..max across draws and the number of draws whose CI lies <0 / >0):
 B1 -> B1+G / +P in the main LOCO test and in the extrapolation splits (leave-one-n-out; MNIST also leave-one-wd-out and Adam->SGD), ridge and GBM, full/compact8/compact4 sizes; direct contrast R2(B1+G) - R2(B1+P_draw) with CI ("GvsP");
 B0 -> B0+G / +P (full 18 and compact4) in LOCO at all cutoffs (design-proxy question; the early cutoffs 10/30/100 are the point).
### A5.B Hard-case subsets (EXPLORATORY: chosen by outcome / by design)
Subsets (per target): (1) hard tercile = top third of configs by mean (over the five cutoffs) config-level MSE of the ridge LOCO B1 prediction (outcome-selected); (2) the complement; (3) label noise > 0; (4) off-grid wd block E2 (MNIST only);
(5) no label noise. Evaluated (a) as OOF predictions of the full-set LOCO restricted to the subset (R2 about the subset mean) and (b) refitting LOCO inside the subset (ridge, subsets with >=8 configs). B1 vs B1+G, B1+compact4, B1+compact8 and,
because selecting on B1 error favours any added regressor (regression to the mean), the RP/NZ placebo blocks (mean of 3 draws) as controls. CIs: config bootstrap within the subset.
### A5.C Fashion GBM gap (step6_C.py; B1 vs B1+G, +compact4, +compact8; target gap_15kw, cutoff 1000)
 (1) block-restricted permutation (G rows permuted among configs inside block A/B, 300) and free permutation (300), GBM refit each time, one-sided p=(#null dR2 >= obs +1)/(n+1);
 (2) full-pipeline refit bootstrap (300): resample configs, LOCO with all copies of the held-out config removed, GBM refit on the multiset of training copies for B1 and B1+X, weighted dR2; percentile CI;
 (3) plain bootstrap (10000) for reference.
### A5.D Compact geometry extrapolation tables
dR2 [95% CI] for compact4, compact8 and full G, ridge (GBM also stored), per split (LOCO, leave-one-n-out, leave-one-wd-out, Adam->SGD) x target x cutoff, pooled (and per fold in the CSV), MNIST and Fashion.
### A5.E Within-config signal in accuracy points (step6_E.py)
Target y - config mean (config mean over the seeds of the held-out config itself), features centred per config, ridge LOCO. RMSE x100 ("points") of: null (predict 0), B0c, B0c+G, B0c+compact4; gain = RMSE(B0c) - RMSE(B0c+G) with config
bootstrap CI; "explained SD" = sqrt(MSE(B0c) - MSE(B0c+G)) in points. Note the residual about a 3-seed (MNIST A/B) or 5-8-seed config mean is deflated by sqrt((m-1)/m) relative to the ddof=1 within-config SD (also reported).

## Amendment 6 (review 4, post hoc) -- 2026-10-16, written AFTER the confirmatory results (confirm/results/) and review4.md were seen
Honest status: every item below is post hoc and exploratory unless it re-reports a pre-registered number. None changes the pre-registered classification of any confirmatory cell. step7_confirm.py is imported, not modified (sha256 1932e941... verified unchanged).
Code = step8_common.py, step8_a_subsample.py, step8_a_conf.py, step8_bcd.py, step8_e.py, step8_summarize.py; outputs = analysis_v2/ext5/ (SUMMARY.txt has all numbers). CPU only, <=4 processes, BLAS threads capped (step4_lib pins 2), loadavg < 60.
Learner, targets (acc_15kw, gap_15kw), cutoffs, LOCO, 10000-draw config bootstrap (default_rng(0); B reproduces primary_ridge.csv deltas and p+ exactly) as in the frozen pipeline.
### A6.A Estimation-error explanation
 (1) Old set (ii) (305 runs, 69 configs, old B1 with is_sgd/log10_steps): 480 random draws of 28 configs (LOCO training size 27), plus 150 draws at 14 and 49 and the full 69; ridge LOCO dR2 for B1 -> B1+G, +compact4, +NZ18 (noise_block), +RP18 (ext4 placebo_feats, draw d mod 5). Distribution per cell (mean, quantiles, fraction<0); confirmatory dR2 located in the old-27 distribution (A_conf_vs_old27.csv).
 (2) Confirmatory set: (i) within-architecture LOCO (14 configs each) and a pooled interaction-aware B1x = B1 + is_cnn x B0; (ii) 1-2 dimensional blocks: te_fisher, te_eff_rank_pr, te_mag_t1, te_h1_top5, fisher+eff_rank, in-fold PCA-1/PCA-2 of G, in-fold PCA-1 of compact4; (iii) nested alpha over logspace(-4,6,21) and fixed alpha in {.1,1,10,100,1e3,1e4}, with the alpha chosen by the default nested CV recorded.
### A6.B Harm p-values (post hoc)
 One-sided p=(#{d*>=0}+1)/(NB+1) for H0 dR2>=0, Holm over the 10 cells within family G and within family compact4, reported beside the pre-registered p+ (dR2>0) Holm values from primary_ridge.csv.
### A6.C Placebo contrast (post hoc)
 Bootstrap of the mean over the 5 draws of R2(B1+G)-R2(B1+P_d), draw-averaging inside every bootstrap rep (shared config resamples), P = RP18 and NZ18; percentile 95%/90% CIs; Holm over 10 cells on the one-sided p (G not worse). Also the draw-mean dR2 of the placebo versus B1.
### A6.D Precision
 90% CI half-width per cell (G, compact4) from the confirmatory bootstrap; plug-in simulation holding the 28 configs' OOF predictions fixed: sample K in {28,56,112,224,448} configs with replacement, inner bootstrap (600) of that sample, 150 outer reps, mean half-width; power-law fit in K for the K giving half-width < margin (.03 acc / .05 gap). Assumes the true increment is 0 and per-config loss contributions stay exchangeable; does not model the reduction of estimation harm with K.
### A6.E Non-geometric placebo
 Inspected run directories, configs and caches: no weight/gradient/activation statistics exist for confirmatory or set (ii) runs (log_opt_diag false; only 300 side-experiment runs have optimizer diagnostics; model_final.pt is end-of-training only = leakage). Built CV18 from run/* scalar curves only (6 curves x {value at cutoff/10, last-5 slope vs log10 step, std}), same cutoff snapshot as G, and ran it as an 18-column placebo with the same LOCO ridge and bootstrap (Holm over 10 cells). It is a training-curve block, not a weight-norm block; the paper must say so.

## Amendment 7 (review 5, post hoc) -- 2026-10-16, written AFTER confirm/results/, ext5/ and review5.md (W1, W3) were seen
Honest status: all items are post hoc and exploratory; none changes the pre-registered classification of any confirmatory cell. step7_confirm.py imported, not modified. CPU only, 4 worker processes, BLAS threads capped (step4_lib pins 2), load < 45.
Code = step9_common.py, step9_a_power.py, step9_b_within.py, step9_c_lowdim.py; outputs = analysis_v2/ext6/ (A_power_summary.txt, A_power_mde.csv, A_power_summary.csv, A_power_reps.csv; B_within_config_confirmatory.{txt,csv}; C_lowdim_table.{txt,csv}, C_lowdim_cells.csv).
### A7.A One-sided power / minimum detectable gain (confirmatory set, 28 configs)
 Per cell (2 targets x 5 cutoffs), step5_power geometry-shaped injection: s = z-scored te_fisher (or te_eff_rank_pr) residualised on [1, standardised B1]; beta = OLS(y ~ B1), e = y - fit; per rep e is permuted among configs (same permutation in all cells of a rep, rng seeded by rep, so cross-cell dependence of residuals is kept), y0 = fit + e_perm, y_s = y0 + b*s with b from step5_power.solve_b so that s explains incremental R2 r in {0, .01, .02, .03, .05, .10} (+ .15, .20 to bracket the MDE).
 Tests exactly as step7: ridge LOCO nested alpha, B1 vs B1+X with X = full G (18), compact4 (both contain the injected feature), and the 1-dim block = injected feature alone; 10000-draw config bootstrap; pre-registered one-sided p+ = (#{d*<=0}+1)/10001; unadjusted per cell and Holm over the 10 cells. 200 reps per level and signal (all 10 cells in each rep): the 80% power crossing has binomial SE <= 2.8 pts at 200 reps, and the full job (2 signals x 8 levels x 200 reps x 10 cells x 4 LOCO fits) took ~25 min on 4 CPU processes, so no reduction was needed.
 MDE80 = smallest r with detection >= 0.80, linear interpolation between grid levels; nan if not reached by r = .20.
 Optimism caveats: (1) the injected signal lies exactly in the tested block (a linear effect of one observed feature that is in the block), the best case for detection; (2) the signal is homogeneous across configs and cutoffs and linear, with no measurement noise in the feature beyond what it has; (3) the null residual is the empirical B1 residual permuted among configs, so it keeps the real clustering but its structure is the in-sample OLS fit, which is optimistic about B1 and makes the residual too homogeneous; (4) injected r is an in-sample incremental R2 for a z-scored feature; the observed LOCO dR2 is smaller (see mean obs dR2 in the output); (5) the 28-config bootstrap is slightly anti-conservative at r = 0 (unadjusted size above .05 in several cells, up to .12 for 1-dim blocks); Holm size is <= .04 per cell; (6) cells are simulated with shared permutations but the injected effect is the same r in all 10 cells, which favours Holm.
### A7.B Within-configuration replication
 Centred within-config analysis of step6_E / step4 3a on the 84 confirmatory runs (28 configs x 3 seeds): target y - config mean, B0 and G centred per config (design columns drop out), ridge LOCO, B0c vs B0c+G and B0c+compact4, cutoffs 10..1000, acc_15kw and gap_15kw; 10000-draw config bootstrap (95% CI); dR2c = (SSE(B0c)-SSE(X))/SS(centred target) as in step6_E; RMSE in accuracy points; centred N(0,1) 18-column control (5 draws); one-sided p+ and Holm over 10 cells per model.
### A7.C Promoted low-dimensional table
 B1 vs B1+X for X in {te_fisher, te_eff_rank_pr, te_mag_t1, PCA-1(G, in-fold), compact4, full G, CV18, noise18}; reuses ext5/A_conf_ii_lowdim.csv, confirm/results/primary_ridge.csv and ext5/E_curve_block_placebo.csv; only noise18 (draw-mean of 5 draws inside each bootstrap rep) is recomputed because ext5 holds only its 95% CI. Summary: mean dR2 over 10 cells, cells with 90% CI upper bound < 0 / lower bound > 0, maximum cell dR2 with 90% and 95% CI.
