# NN_RATIO_CHECK: does the test->nearest-train NN ratio track the recon gap beyond N?

Script: /tmp/claude-2292/s/nn.py (CPU, 8 threads). Runs: MLP d16 AE, 4 N (250/1000/4000/full) x wd (0, 1e-4) x 3 seeds = 24 per dataset.
nn = final `nn_ratio` from analysis/{,fashion/}memorization/*.npz (median test->nearest train probe latent / median train->NN train latent; ae_memorization.py).
y = log(final_recon_test/final_recon_train) (same as ae_findings_stats.py section 3b). PCA ref = same ratio on pixel-PCA-16 fit on the N-pool, same probes.

## Verdict
CONFOUNDED / N=250 ARTEFACT. No evidence of a relation beyond N, and the NN ratio is mostly the data-subset geometry (PCA ref explains it).
- The -0.96 / -0.87 partials come from the adjustment (linear in log N + wd) being too rigid: y vs log N is strongly nonlinear (y = 6.1, 1.4-1.7, 0.4, 0.01) and nn_ratio is a step function of N (about 0.8-0.87 at N=250, about 1.0 elsewhere). N=250 is a leverage point that the linear-logN fit cannot absorb, so the residuals of nn and y are both driven by it.
- Replace the log-N term with N dummies (flexible N): partial(nn, y) = -0.29 (Fashion) / -0.07 (MNIST). Drop N=250: Fashion -0.38 (n=18; -0.12 if full also dropped), MNIST +0.35 (wrong sign).
- Original FINDINGS (|r|<=0.2) used log N + family + wd on a differently-assembled set and I could not reproduce its exact table; with the 3-covariate adjustment used here MNIST gives -0.87/-0.84 (N<=4000) exactly as the Fashion stats file reports. So the two analyses differ in adjustment, not in data; neither is a clean test (only 4 N levels).

## Results (Fashion | MNIST)
1. Reproduced: partial(nn,y|logN,wd) all24 -0.957 | -0.873; N<=4000 -0.922 | -0.843; excl N=250 -0.384 | +0.350; excl 250 and full -0.116 | +0.546.
2a. Within-config across seeds (config means removed, pooled, permutation p): all n=24 r=-0.18 (p=0.45) | +0.03 (p=0.91); excl N=250 -0.54 (p=0.03) | +0.27 (p=0.45). Per-config r (n=3 each) have inconsistent signs on MNIST; Fashion mostly negative but the pca-ref gives the same (-0.55 vs -0.54), see 2e. 
2b. Leave-one-N-out (partial): drop 1000: -0.95 | -0.86; drop 4000: -0.95 | -0.90; drop full: -0.92 | -0.84; drop 250: -0.38 | +0.35. Only dropping 250 changes anything.
2c. = row "excl N=250" above.
2d. Time-resolved (119 snapshots/run): level Spearman(log gap, nn) is strongly negative at N=250 (-0.9 | -0.85) but this is a shared monotone time trend: correlation of first differences is ~0 (mean -0.01 | -0.01; per-config |r|<=0.05), partial on poly2(log step) mean +0.06 | -0.05 (sign random across runs, 61% | 39% positive). The NN ratio drifts down late (N=250: 0.96->0.79 Fashion, 1.02->0.91 MNIST) while the gap is already open; it does not track gap fluctuations.
2e. PCA reference ratio (no network): N=250 0.921 | 0.970, N=1000 0.992 | 0.988, N=4000 0.992 | 1.002, full 1.007 | 0.993. Already below 1 at N=250 (sample-subspace effect), so about 0.13 (Fashion) / 0.10 (MNIST) of the latent 0.79 / 0.87 is the train-fitted-subspace reference; the network excess below PCA is -0.13 / -0.10 at N=250 only (nn is within noise of pca at N>=1000). partial(pca_ref, y) = -0.72 | -0.50 with the rigid adjustment (same artefact: reference has no network yet "predicts" the gap); nn-pca still -0.93 | -0.74 rigid but -0.18 | +0.16 with N dummies. Within-config r(nn,pca)=0.71 | 0.39: seed variation in the subset geometry feeds nn.

## Caveats
Only 4 N levels, 3 seeds; N=250 gap noise large. A residual N=250 excess (latent NN ratio below PCA ref by 0.13/0.10) is real but is one level, confounded with the largest gap and with log-N misfit, so it cannot be separated from N.
Verdict wording for FINDINGS_fashion item 3: change "nn_ratio strong on both" to "N=250 leverage artefact of the log-N adjustment; null with flexible N, within seed, and in time-resolved first differences".
