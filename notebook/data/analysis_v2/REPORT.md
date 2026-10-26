# v2 analysis report (2026-10-15): 123 runs (105 MNIST blocks A+B, 18 CIFAR-10 ResNet-18)
Code: /homes/lbastian/latent-dynamics/analysis_v2/ (ANALYSIS_PLAN.md incl. amendments; run_all.sh 6).
Outputs: this dir. figures/*.png, step2_*.csv, step3_results.csv, metrics_long.csv, runs.csv, step2_summary.txt, step3_summary.txt.
Cheap metrics (Fisher, norm, probe acc, eff. rank) on all ~120 snapshots; topology/magnitude on ~66 per run
(all steps ≤10, every 2nd after, plus last). Full 1000-point probe sets. Mag(t) fixed 19-pt grid; PMag fixed window [0.5,10]. No pinv NC1.

## Step 2 (descriptive)
- Slingshots: 15/105 runs, all wd=0. Mid-transition at 15k only for mnist|nfull|wd0|ln0.2.
- (a) Train−test divergence vs accuracy gap; log10 half-progress time(Δ_metric) − time(acc gap), negative = metric diverges first.
  Fisher +0.65 [+0.38,+1.00] (n=26, lags gap in 85%); eff. rank −1.19 [−1.44,−1.06] (n=17, leads in all);
  probe acc −0.14 [−0.95,−0.02]; Mag(t=1) −0.91 [−2.05,+0.01]; H1 top-5 −1.15 [−2.06,+1.20]; PMag −0.09; H1 total +0.14; raw norm +0.21 (CIs incl. 0).
  Train−test probe-acc divergence ~0.15 at n=250 → ~0 at full; negative under label noise (train measured vs noisy labels).
- (c) Geometry vs test acc, median log10 t50 difference (geometry − test acc): probe acc −0.29 (geometry first 100%), eff. rank −0.72 (100%),
  Mag(t=1) −0.34 (87%), Fisher +0.46 [+0.17,+0.62] (12%), H1 top-5 +0.71 [+0.29,+1.16] (11%).
  Caveat: probe acc / eff. rank move immediately from init (start-state effect). Fisher and H1 top-5 move after test accuracy.
- (b) Phase grids: fig2b_grid_wd.png, fig2b_grid_ln.png (+CSVs).
- (d) Redundancy at step 5000 (35 configs, Spearman with Fisher): H1 total −0.89 [−0.97,−0.72]; H1 entropy −0.85; probe acc +0.89;
  Mag(t=1), PMag −0.79; H1 top-5 +0.18 [−0.24,+0.55] (non-redundant, but uncorrelated with test acc +0.04 and gap −0.20).
  Fisher vs test acc +0.88, vs gap −0.84.
- CIFAR (6 configs): train−test signs match MNIST. Fisher, probe acc, Mag(t=1) diverge as gap opens; gap grows with noise
  (n10000: 0.16/0.33/0.51). Eff. rank Δ negative at high noise. CIFAR Fisher Δ at nfull ln0 = +5.9 vs matched-MNIST median +0.37.
  Matched-gap bins thin (1–2 MNIST configs). CIFAR timepoints = fractions of training (5/25/50/100%).

## Step 3 (early prediction; leave-one-config-out, 105 runs / 35 configs)
B0 = baseline curves; B1 = B0 + design (n, wd, label noise); G = geometry. ΔR² with 95% config-bootstrap CI.
- Test acc @15k, ridge, cutoffs 10/30/100/300/1000: R²(B1) ≈ 0.94–0.95; ΔR²(B1→B1+G) = −0.04 [−0.07,−0.02], −0.03, −0.02, −0.02, 0.00 [−0.02,+0.01].
  Pre-slingshot target ≈ 0 to −0.01. GBM same.
  B0→B0+G: +0.47 [+0.29,+1.03] at cutoff 10, +0.20 at 30; vanishes by 100 (geometry proxies design vars).
- Gap @15k, ridge: R²(B1) 0.59–0.66; ΔR² −0.11 [−0.22,−0.00], −0.03, +0.01, 0.00, +0.09 [−0.07,+0.33] at cutoff 1000. GBM +0.28 [−0.05,+0.75] at 1000.
  Permutation p 0.007 at 1000, 0.037–0.047 at 100/300 (only vs shuffled geometry). G alone R² 0.77 ridge / 0.81 GBM at 1000.
- Gap pre-slingshot: ΔR² −0.11 to +0.01; +0.01 [−0.07,+0.14] at 1000.
- Verdict: no reliable increment over curves+design for test acc; for gap at most a weak, non-significant gain at cutoff 1000.
- Sensitivity: excluding onset ≤15k runs changes nothing (200 perms, ridge only). Exploratory peak-test-acc (10 label-noise configs): inconclusive.

## Deviations from plan
Amendments recorded in plan; CIFAR-vs-MNIST matched-gap comparison added; run discovery restricted to _phase_n/_gap_n/_rn18_n;
no target/feature changes after seeing outcomes.

# UPDATE 2026-10-15 (afternoon): audit + extension (blocks C, D, E2) — supersedes "pre-slingshot" numbers above
Full details: ext/{set_i,ext,ext_adam,sgd,audit}/step3_summary.txt, step2_summary.txt; audit in ext/audit/.
## Audit: no leakage/bug. PASS: features ≤cutoff; no target leakage; LOCO scaling + nested alpha; baseline fairness; alpha grid; GBM hparams;
bootstrap (+ per-config Wilcoxon p=0.13–0.63); delta-target; negative controls (noise cols, permuted G, shuffled target).
Positive control (noisy target copy) detected. POWER: min detectable dR2 ≈ +0.03 (acc; R2(B1)≈0.94 ceiling), ≈ +0.15–0.25 (gap; CI ±0.2).
=> acc null robust; gap null = "cannot detect small gains". Permutation p is a weak null (shuffled G = noise); prefer CI/sign.
Compact geometry sets (2–8 feats) do not change conclusions; only marginal: H1-top5 on gap_15kw cut100 +0.027 [+0.000,+0.069].
## Extension (305 MNIST runs, 69 configs; design adds is_sgd, log10_steps; targets: 15k point/window(12–18k median), final window, pre-slingshot (floor onsets, any/N-type))
- Set i (orig 35): reproduces; window = point. gap_15kw cut1000 +.075 [−.070,+.307]; gap_pre cut10 −.112, cut1000 +.009.
- Set ii (69 configs): R2(B1) acc .95–.97, gap .71–.76. acc dR2 ≤ +.013 (tiny, GBM ≤0 → negligible). gap_15k cut1000 ridge +.042 [−.033,+.152],
  GBM +.195 [−.006,+.497] (GBM gap cut1000 +.17–.30, CI lower ≈0). gap_finw cut1000 +.013. pre-slingshot −.025..+.007.
- Set iii (Adam, 63 configs): same pattern; gap_15kw cut1000 +.063 [−.032,+.203].
- Set iv (SGD, 6 configs): underpowered; acc≈gap (train acc saturates).
- G alone R2 0.77–0.84 at cut1000 vs B1 ~0.6–0.7 (gap): geometry is a good proxy for design+curves, not additive beyond them.
## Step-2 with 69 configs: Fisher lags acc gap (+0.67 [+0.51,+0.93] log10) and test acc (+0.10 [+0.05,+0.17]);
eff. rank (−1.17 [−1.36,−1.07]) and Mag(t=1) (−1.03 [−1.71,−0.93]) diverge before the gap; probe-acc lead vanishes (−0.03 [−0.06,+0.03]);
H1-top5 vs gap +0.80 [−0.89,+1.49]. Redundancy @5000: H1 total vs Fisher −0.94; H1-top5 +0.14 (non-redundant, uninformative); Mag(t=1) −0.76; PMag −0.82.
SGD: Fisher lags +0.22; rank, Mag(t=1) lead.

# UPDATE 2 (2026-10-15 evening): reviewer analyses, Amendment 3 (outputs ext2/mnist, ext2/fashion)
1 Equivalence (margins ±.03 acc, ±.05 gap; 90% CI): ridge B1→B1+G equivalent 16/20 cells, positive 2 (acc, ≤+.006, inside margin), inconclusive 2 (gap cut1000: +.036 [−.033,.138], m*=.119). GBM mostly inconclusive; gap cut1000 +.170.
2 Extrapolation (leave-one-n-out, leave-one-wd-out, train Adam→test SGD): G helps only at cutoffs 10–30 where B1 is poor; at ≥100 mostly hurts (LONO cut1000 acc −.188, gap −.115). G alone unstable (R² down to −12.5).
3a within-config seed residual (SD ~.0065): ΔR² +.155 [.06,.28] (acc) / +.168 [.06,.30] (gap) at cut100; +.067/.061 at 300; ~0 at 1000.
3b design residual: linear +.2–.3 (acc) mostly nonlinear design structure; quadratic-design residual ≤+.06 (acc cut1000 +.037 [.01,.13]; gap +.053 [.02,.16]).
4 Curve-extrapolation baseline B2 (power/exp/log-linear): no better than B1 (worse for gap); B2→B2+G acc equivalent; gap cut1000 +.155 [.003,.384] (partly degraded base).
5 Power sim (60 reps): acc cut100 min detectable ΔR² ≈ .011; acc cut1000 >.013 (G block costs ~−.010); gap: not detected even at ρ=.7 (ΔR² +.06 cut100, +.11 cut1000) → gap increments <~.1 unresolvable.
6 Timing: Fisher lag, eff-rank lead, Mag(t=1) lead robust to q=.25/.5/.75 and slingshot exclusion (51/69 configs); H1-top5, probe acc, PMag, H1 total sign-unstable. Undefined timing = non-monotone Δ curves, non-random (label noise / large n).
7 Redundancy with Fisher at steps 300/1000/5000/15000: H1 total −.98/−.97/−.94/−.95; Mag(t=1) −.75/−.83/−.76/−.80; PMag −.91/−.87/−.82/−.78; H1 top-5 −.02/+.37/+.14/+.25.
8 PMag window ρ: MNIST −.926 [−.98,−.57], Fashion −.902 [−.95,−.56], CIFAR −.128 [−.83,+.66] (27 runs, 9 configs each).
9 Fashion-MNIST replication (105 runs, 35 configs): ridge acc CIs include 0 except cut30 (+.020 [−.001,.054]); gap ridge lower bounds ≤0 everywhere (cut1000 +.037 [−.09,.16]); GBM gap cut1000 +.254 [.034,.546] (non-replication for GBM). LONO: G hurts. 3a: cut10 +.39/.36, cut1000 −.07.

# UPDATE 3 (2026-10-15 night): review-2 analyses, Amendment 4 (outputs ext3/{mnist,fashion}, code step5_*.py)
Ridge B1 vs B1+G unless stated. m* = smallest margin at which the 90% CI is equivalent.
1 Margin sensitivity (10k config bootstraps): MNIST ridge acc m* .008–.026 (all <.03); gap m* .021–.045, except cut1000 .117.
  RMSE translation: 0.5 pt RMSE ≈ dR2 .016–.021 (acc, RMSE 1.6–1.9) / .049–.055 (gap, RMSE 4.6–5.1); ±.03 acc ≈ 0.75 pt, ±.05 gap ≈ 0.5 pt.
  MNIST GBM m* acc .018–.062, gap .035–.39. Fashion ridge m* acc .040–.079, gap .087–.143 → no Fashion cell equivalent at the stated margins.
2 Design-aware uncertainty: block-stratified bootstrap ≈ plain. Full-refit bootstrap (400) is 2–10× wider: acc equivalent at ±.03 only at cut10 (m* .024; others .034–.063); gap m* .13–.19.
  Block permutation p vs free: matters mainly at gap cut1000 (.084 vs .004).
3 Multiplicity (2 targets × 5 cutoffs): MNIST TOST equivalent 9/10 unadjusted, 5/10 after Holm (acc 10/30/100, gap 10/30). Fashion 0/10. Window = point targets.
4 Learners/feature sets (extrapolation cells, cut ≥100, CI upper <0): full G hurts (LONO ridge 6/6; kernel ridge, MLP 4/6; Fashion 5–6/6). compact4/pca2 mostly neutral (LONO ridge compact4 3/6, LOWO pca2/pca4 0/6), with smaller harm. PCA-4 not safe (gap −.88).
  Main LOCO: acc flat for all learners; gap cut1000 +.10 to +.20 for kernel ridge/MLP (same signal as GBM).
  Extreme negative R²: mostly low held-out variance (n=full gap SD 2.7 pts); G-alone −95/−39 (n=250, cut100) is real failure (RMSE 68/44 vs 2.6/3.0). Median RMSE ratio B1+G/B1 = 1.05.
5 Residual targets (B0→B0+G): linear design +.07 to +.31 (acc); quadratic ≤ +.062; GBM ≤ +.10 (wide CIs); config-mean +.155/+.168 at cut100, ~0 at cut1000. Fashion: only within-config cut10 (+.39/+.36).
6 Power v2 (MNIST, 200 reps, geometry-shaped signal): 80% power at dR2 ≈ .016–.017 (acc), ≈ .04–.075 (gap). FPR ≤ .5%. Optimistic (no unmodelled design nonlinearity in the noise).
7 Learning-curve baselines (isotonic gap, last-3 median, slope): never help gap (ridge −.056 to +.007), acc ≤ +.004. G on top: acc ~0, gap cut1000 +.03 to +.10.
Verdict: fixes 1–7 answered. Wording to use: margins chosen post hoc, justified as 0.5–0.75 RMSE points; MNIST 9/10 → 5/10 after Holm; Fashion 0/10, "does not contradict" only; "unrestricted geometry hurts extrapolation, low-dimensional geometry mostly neutral"; gap increments below ~0.05 unresolved (with the optimism caveat). Scope (fix 8) needs new data.

# UPDATE 4 (2026-10-16): review-3 analyses, Amendment 5 (outputs ext4/{mnist,fashion}, code step6_*.py)
A Placebo blocks (random 1-D projections of the same latents, and Gaussian noise; sizes 18/8/4): neutral in LOCO, like G. Under extrapolation G hurts far more than any placebo of its size (LONO acc cut1000 G −.188 vs RP −.006; gap cut300 −.222 vs −.007), so the harm is not just a feature-count effect. G is a better design proxy than placebo early (B0→B0+X acc cut10 +.71 vs RP +.42 vs noise −.12).
B Hard-case subsets (outcome-selected, exploratory): no robust ridge gain. GBM gap cut1000 +.21 to +.24 in the MNIST hard tercile (CIs touching 0), +.35/+.39 in the Fashion hard tercile; placebos ~0.
C Fashion GBM gap cut1000 +.254: block permutation p .037 (compact4 .017), but the refit bootstrap includes 0 (+.113 [−.116,+.443]) → suggestive only.
D Compact extrapolation: compact4 is least harmful and sometimes neutral (LOWO at cut ≥100), but LONO gap cut1000 is still −.187; compact8 and full G reliably hurt. Do not say "compact is neutral".
E Within-config: G lowers seed-level RMSE by ~0.05 pts (cut100) and 0.02 pts (cut300) of accuracy; explained within-config SD 0.14–0.22 pts. Real but tiny. Fashion: only at cut10.
numbers_check3: no numeric errors; 5 wording FAILs (compact "mostly neutral" in abstract; extrap caption at cut10–30; 9/10 counts a P cell; ±.03 ≈ 0.7–0.9 pt not 0.75; KRR/MLP ordering).
User decision 2026-10-16: CPU MNIST confirmatory replication (random configs, second architecture, frozen margins). No CIFAR grid.

# UPDATE 5 (2026-10-16): pre-registered confirmatory replication (confirm/, PREREG sha e4fe0682…, step7 sha 1932e941…)
84 runs = 28 random off-grid configs × seeds 100–102; 14 MLP / 14 CNN; 0 failures. Results in confirm/results/summary.txt.
Off the ceiling: R²(B1) is .77–.82 (acc) and .67–.71 (gap), against .95–.97 / .69–.76 on the old grid.
Primary (ridge, full G): equivalence confirmed 0/10 and positive 0/10. Point dR2 is −.026 to −.127 in every cell, and the 90% CI lies wholly below 0 in 5/10 (acc cutoffs 300/1000, gap cutoffs 30/300/1000). So full geometry HARMS the ridge forecast on new configs.
compact4: equivalence confirmed 1/10 after Holm (gap cut10; unadjusted 4/10), positive 0/10. Point estimates −.007 to −.09, with +.020 only at acc cut100.
GBM (descriptive): mostly inconclusive or equivalent. Full G: acc cut1000 +.043 [−.03,+.19], gap cut1000 +.014. No reliable gain.
Gap at cut1000: ridge G −.094 [−.17,−.03]; GBM G +.014 [−.06,+.11] → the earlier GBM gap-1000 signal does not replicate.
Placebo: G is worse than random-projection or noise blocks of the same size (G−placebo at cut1000 ≈ −.06 to −.08).
Transfer old→new: ridge mostly inconclusive or harmful for full G, compact4 often equivalent; scattered GBM positives (e.g. mlp_only compact4 acc cut30 +.31 from a base with negative R²), descriptive only.
Per architecture: harm in both MLP and CNN.
Verdict: no configuration of the test finds an increment. The pre-registered equivalence claim is NOT confirmed, because geometry tends to harm rather than because it helps.
2026-10-16: dense optimizer-logging flags (optdense.py) merged into the main tree after the confirm extraction finished. The frozen copies of train.py and config.py are in confirm/*.prereg_frozen.

# UPDATE 6 (2026-10-16): review-4 analyses, Amendment 6 post hoc (ext5/, step8_*.py; SUMMARY.txt has all numbers)
A Estimation error: on the old grid subsampled to 28 configs, dR2 is centred near 0 but wide (5–95% about ±.13 acc, ±.24 gap). Full G mean −.015, with 56% of draws below 0. The confirmatory −.073 sits at the 5–38th percentile, so it is plausible but on the low side. Within architecture the harm persists (MLP −.069, CNN −.117). Alpha grid ruled out as the cause. 1–2-dim geometry blocks: harm is about −.01, with no gain.
B Post-hoc harm p (Holm): full G 3/10 (acc 300, acc 1000, gap 300), compact4 2/10. Pre-registered p+ Holm = 1.0 in every cell.
C Placebo contrast (draw-mean bootstrap, Holm): G vs RP18 negative in 10/10, significant in 2/10. vs noise: 1/10. The placebos themselves lower R2 (noise 10/10).
D Precision at 28 configs: full-G 90% half-width .06–.16, so it could not confirm equivalence in any cell even at a true 0. Needs about 115 configs (median). compact4 could pass in 4 cells.
E There are no weight or gradient norms for set (ii) or confirm runs. A non-geometric 18-col curve-statistics block (CV18) harms as much as G (mean −.088 vs −.073). So the harm is mostly a column-count effect at 27 configs, not specific to geometry.
Headline: no detectable gain at the configuration level. The confirmatory losses are generic to adding ~18 columns at 27 training configs; geometry is not shown to be uniquely harmful.

# UPDATE 7 (2026-10-16): review-5 analyses, Amendment 7, post hoc (ext6/, step9_*.py)
A One-sided power on the confirmatory set (signal injection, 200 reps): MDE80 per cell, median (Holm). Full G .14 (.10–.20; not reached in 2/10 cells); compact4 .095; 1-dim te_fisher .087. Any-cell detection at a true +.05: full G 71%, compact4 98%, 1-dim 99%. Gains of ≤.03 would be missed. Optimistic.
B Within-config signal does NOT replicate on the confirmatory runs (cut100 acc +.010 [−.014,+.125], RMSE gain .006 pts; all Holm p+ ≥ .44).
C Low-dim blocks on the confirmatory set: mean dR2 −.014 to +.007; no block has any cell with a lower bound above 0; Holm p+ = 1 everywhere.
numbers_check5: 4 wording FAILs. Do not say "could not have been confirmed"; say "could not be with the observed widths". "As much as non-geometric columns" holds only for CV18 (G is worse than RP in 10/10 cells, 2/10 significant). The subsampling spread mislabelled envelopes as cells. The low-dim lower bound is +.0006, not +.003.
User decision 2026-10-16: second frozen confirmatory study on Fashion-MNIST, CPU.

# UPDATE 8 (2026-10-16): pre-registered Fashion-MNIST confirmatory replication (confirm_fm/, PREREG sha 65c25a9e…8c9e frozen 14:19:51Z, combined PREREG+step10 sha b0ac0bf8… recorded 14:20:05Z)
90 runs = 30 random off-grid configs × seeds 200–202; 15 MLP / 15 CNN; 0 failures; CPU, queue 14:20:45Z to 20:59:52Z. Results in confirm_fm/results/summary.txt (+ primary_ridge.csv, secondary_*.csv, exploratory_per_arch.csv). All runs log optimiser/weight diagnostics (opt_*), giving a genuine non-geometric block OPT18 (secondary e). Hashes of PREREG, step10_confirm_fm.py and confirm_fm_gen.py recomputed and match. Paper tables/stats: paper_v2_figs/make_tables6.py -> paper_stats6.txt.
Pre-specified headline (differs from MNIST): one-sided positive increment of full G (family F, Holm over 10 cells); equivalence is secondary. The PREREG stated in advance (from MNIST ext5/ext6) that full-G equivalence was not confirmable at K=30 and that only increments of about .10-.15 R2 were detectable at 80% power.
Baseline is much weaker than MNIST: R2(B1) .573–.721 (acc) and .010–.223 (gap; .010 at cutoff 300), against .77–.82 / .67–.71 on MNIST confirm.
Primary (ridge, family F, full G): positive increment 0/10 (Holm; unadjusted 0/10), equivalence confirmed 0/10. Point dR2 negative in 8/10 cells (−.083 to −.009), positive in 2 (acc cut1000 +.045 [−.105,+.238]; gap cut300 +.018 [−.195,+.253]); 90% CI wholly below 0 in 4/10 (acc 30, 100; gap 10, 30). Min Holm p+ = 1.0. 90% half-width .042–.172 (acc) and .072–.313 (gap), every one above its margin, so equivalence was unattainable at the observed widths.
Family C (compact4): positive 0/10 Holm, but 3/10 unadjusted (acc 30 +.015, acc 300 +.048, gap 300 +.105; Holm p+ .39/.17/.32, min .172); equivalence 0/10 after Holm (1/10 unadjusted: gap cut30, TOST .018, Holm .182). Point dR2 −.022 to +.120 (negative in 4/10); no 90% CI wholly below 0. Contrast with MNIST confirm: G negative in 10/10 with 5/10 CIs below 0; compact4 equivalent in 1/10 after Holm.
(b) GBM (descriptive): full G equivalent 2/10, positive 0/10 (dR2 −.138 to +.160); compact4 equivalent 3/10, positive 1/10 (acc cut1000 +.267 [+.022,+.628], p+ .030 unadjusted, from a weak baseline R2 .48).
(c) Gap at cutoff 1000 (hypothesis stated in PREREG): not confirmed for any learner/block: ridge G −.077 [−.425,+.201] p+ .66; ridge c4 +.120 [−.003,+.351] p+ .053; GBM G +.129 [−.066,+.248] p+ .20; GBM c4 +.142 [−.021,+.238] p+ .089. Estimates lean positive for compact blocks (R2(B1) ridge .19, GBM .37); the exploratory Fashion GBM gap +.254 stays "suggestive", neither confirmed nor refuted.
(a) Transfer old->new: 10/80 cells positive (MNIST confirm: 5/80); 9 of the 10 come from baselines with negative R2 (e.g. ridge all G acc cut10 R2(B1) −1.28, dR2 +.743). Only non-degenerate positive: ridge MLP-only c4 gap cut1000 +.107 [+.027,+.261], R2(B1) .64.
(d) Placebos (ridge, mean dR2 over 10 cells): RP18 −.080 (10/10 neg, Holm harm 10/10), NZ18 −.091 (10/10, 10/10), CV18 −.034 (8/10, 1/10); full G −.037, compact4 +.031. G minus RP18 +.043, G minus NZ18 +.053, G minus CV18 −.004: no cell Holm-significant in either direction. This differs from MNIST, where G was below RP18 in 10/10 cells (2/10 Holm). compact4 minus RP18 +.111, positive in 10/10, Holm 7/10 (acc 10, 30, 300; gap 30, 100, 300, 1000); this is wider than the "acc cut10/cut30" in the hand-off note.
(e) Optimiser block: OPT18 vs B1 −.176 (10/10 neg; Holm harm 4/10; 90% CI below 0 in 8/10), OPT4 vs B1 −.024 (8/10; Holm 2/10). G minus OPT18 +.139, G better in 10/10 cells but Holm-significant only in 1 (acc cut10 +.099 [+.053,+.174], where G itself is −.009); G minus OPT4 −.013 (0/10). compact4 minus OPT18 +.207, Holm 9/10; compact4 minus OPT4 +.055, Holm 3/10 (acc 300; gap 100, 300). Reading: an 18-column non-geometric block of weight/gradient/curvature statistics loses MORE than full G; the loss from adding columns is generic. By the PREREG rule "geometry-specific" needs G minus OPT > 0 with Holm p+ <= .05: 1/10 cells for OPT18, 0/10 for OPT4, and only in a cell where G has no increment over B1.
(f) 1-dim blocks (40 cells): mean dR2 +.034 (te_fisher), −.007 (eff rank), +.045 (te_mag_t1), +.029 (PCA-1). 7/40 cells have 90% lower bound > 0, 0/40 after Holm (min Holm p+ .058). The strongest: gap cut1000 te_fisher +.249 [+.115,+.391] (Holm .058), te_mag_t1 +.214 (.074), PCA-1 +.212 (.084). This is the best evidence of a gain in either confirmatory study; it is not confirmed. MNIST ext6 analogue: no cell with lower bound > 0.
(g) Within-config (centred): G dR2c −.206 to +.085 and compact4 −.101 to +.062, no 95% CI above 0 (min Holm p+ 1.0 and .57). Only OPT18 acc cut10 +.169 [+.045,+.303] (Holm .027). So the exploratory within-config signal does not replicate for geometry on either dataset (MNIST ext6 B: min Holm p+ .439).
Exploratory per architecture (15 configs each): pooled R2(B1+G) < R2(B1) in 8/10 cells for MLP and for CNN; CNN is above B1 at cutoff 1000 (acc .400->.576; gap .148->.228) while MLP loses (acc .879->.738). compact4 above B1 in 7/10 MLP and 6/10 CNN cells. Small subsets; hypothesis-generating only.
Deviations to disclose (paper Sec. 6 and App. D): (vi) confirm_fm_gen.py and step10_confirm_fm.py were written and dry-run on 10 OLD Fashion configs (and the OPT plumbing on 400-step throw-away smoke runs) before the PREREG freeze; no confirmatory job file or run existed then, and the script was unchanged afterwards (hash matches). (vii) No external timestamp yet (EXTERNAL_TIMESTAMP_TODO.txt; combined hash b0ac0bf84ed54843fe3b078de0b67ac16e83ccdfeb4b697ec9ca1cf06cf97e9f). (viii) The full-set config (c11) enters log10 n as 50000 (legacy convention). Unlike MNIST, the analysis script was hashed before launch (14:20:05Z vs queue 14:20:45Z).
Comparison. Exploratory Fashion grid (105 runs): 0/10 ridge cells equivalent, GBM gap cut1000 +.254 [+.034,+.542] suggestive (refit +.113 [−.116,+.443]). Confirmatory Fashion agrees on the ridge null (0/10 positive, 0/10 equivalent) and does not confirm the GBM hint (+.129 [−.066,+.248], p+ .20). Versus MNIST confirm: same verdict for full G (no positive increment, no equivalence) but different sign pattern (MNIST all negative; Fashion mixed) and a weaker baseline; compact4 leans positive on Fashion at late cutoffs (unadjusted), mostly negative on MNIST.
Verdict: no pre-registered positive increment of geometry in 20 full-G and 20 compact4 ridge cells (10 per family per study, Holm); equivalence for full G unattainable at these widths; non-geometric blocks (incl. genuine optimiser statistics) lose as much or more than G, so losses are generic to column count. The only hint of gain (late gap, low-dim blocks, Fashion) is unadjusted and unconfirmed.
