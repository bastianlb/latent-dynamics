# Linear-dynamical-system (LDS) analysis of AE latents vs classifier probe latents

Is the latent transformation largely linear? Same analysis as the classifier study (`vfield_decomp.py` partition + generator, `vfield_spectra.py`), run on the AE grid via the thin adapter `ae_vfield_decomp.py` (classifier scripts untouched; code: `ae_vfield_decomp.py`, `ae_lds_load.py`, `ae_lds_report.py`). Raw per-run outputs: `lds/runs/*.npz` (384 files, 28 MB); tables: `lds/table_side_by_side.md`, `lds/table_sections.md`.

## Bottom line (all statements are descriptive; we did not test mechanisms)
1. **Mostly linear per interval, increasingly less so late.** For the MLP AE (d=16) a global affine map z_t -> z_{t+1} explains held-out R1 = 0.97 of the displacement energy early (steps 10-100), 0.84-0.92 at steps 100-1000, and 0.75-0.84 after step 1000 (N >= 1k; N=250 stays 0.94). Adding label-wise affine adds only +0.02 to +0.07 and a kNN nonlinear term on top only +0.003 to +0.016. The remainder (0.12-0.19 late) is not captured by any of our fitted forms (kNN on the residual in z; the light kernel-ridge variant was no better, see caveats).
2. **Same picture as the classifier, with a different N dependence.** Early (10-100) classifier and AE are indistinguishable (R1 0.975 vs 0.97). For the classifier, R1 stays ~0.97-0.98 at all steps for N <= 4k (mid-training dip at 4k), and collapses late only for full N (0.58 MNIST, 0.60 FM). The AE instead loses linearity late for every N >= 1k (0.81 at 1k, 0.75 at full MNIST), but less severely than the classifier at full N. So: the classifier's late non-affine motion is a large-N phenomenon; the AE's is present at N=1k and gets bigger with N (N=250 AE: 0.94).
3. **Linear AE is NOT exactly linear in this sense** (R1 1000+ = 0.39-0.75 MNIST, 0.80-0.86 FM; early 0.90-0.94), i.e. R1 ~ 1 does not hold even where the map z=Wx is linear, so we did not reproduce the "exactly linear" expectation. Plausible reason (not tested): dW depends on x through the 784-d residual, not only on z=Wx, so the per-interval update is not a function of z alone; also R1 drops with N for the linear AE (the opposite sign from MNIST-MLP at small N). The MLP AE is *more* affine than the linear AE after step 100 on MNIST. Treat the linear AE as a negative sanity check on our expectation, not as a validation.
4. **Class-wise affine adds little beyond global affine and labels are not better than label-free groups.** R2-R1 (late) is 0.02-0.07 (labels). k-means(10) on input pixels gives an equal or larger increment: MNIST labels-minus-kmeans -0.002 +- 0.0005 (late, test), FM -0.007 +- 0.001 (labels beat k-means in 0% of FM runs, 17% of MNIST runs). So the class-wise increment is generic piecewise-affine flexibility, not label-specific structure. (Classifier has no label-free control here; its R2-R1 is +0.07-0.08 at full N late.)
5. **Generator structure.** Both setups: early the per-step generator is dominated by expansion (abscissa +0.05-0.08 per step, 10-15 of 16 eigen-directions expanding) with trace share ~0.01-0.03; the symmetric-traceless and antisymmetric parts are ~0.5/0.5 (rotation ~ shear, i.e. generic, near a random-matrix split). Late: trace share 0.02-0.16 (AE, falling with N; classifier 0.00 at full N, 0.2-0.25 for small N), abscissa -> ~1e-4, number of expanding directions falls with N (AE 7/16 at full, classifier 6.6-7/16 at full) and ~5-6 complex pairs in both. Antisym share is 0.28-0.42 for the AE (lower than the classifier's 0.35-0.5 early-mid but similar late at full N). Gauge caveat applies (below); we do not interpret rotation shares physically.
6. **Train vs test probe and generalization.** Train-minus-test differences are tiny but systematic: late R1 lower on the train probe by 0.012 (MNIST) / 0.017 (FM), unexplained higher by 0.009 / 0.014 (paired, 18 runs each, se ~0.001-0.004); R3-R2 +0.002 / +0.0015. The classifier FM shows a smaller (-0.006) R1 gap and no unexplained difference. Within (dataset, N, wd) groups (N dummies), late-band R1, R2-R1, R3-R2, unexplained and trace share are **not** significantly related to the final recon gap (|rho| <= 0.2, p > 0.3 on MNIST/FM/pooled for R1, 1-R3; FM trace share rho -0.55 (p=0.008) and antisym share +0.44 (p=0.04) are nominal hits among ~48 tests and do not replicate on MNIST (+0.20, -0.38) — we treat them as not established). The between-N Spearman with the gap is large (R1 +0.70, 1-R3 -0.60) but equals the N effect (rho with log N -0.78): N confound. We did not find an LDS quantity that tracks generalization beyond N; consistent with the dense null test (affine-residual share not an early indicator).
7. **Latent dimension.** R1 late rises with d (MNIST full: 0.51 / 0.75 / 0.91 for d = 8/16/64; unexplained 0.35 / 0.19 / 0.08): the more latent room, the more affine the motion. Not matched to the classifier (no d sweep run).

## Side-by-side (test probe, wd=0, d=16, mean of 3 seeds)
Schedules are identical (120 log-spaced snapshots, steps 0..20000, same dstep per interval), so per-interval R1 is directly comparable. "unexplained" = 1 - R3 (kNN). trace/sym/antisym are shares of ||L||_F^2 (L = logm(A_t)/dstep), medians over intervals in the band. Band = interval end step.

### mnist (test probe, wd=0, mean over 3 seeds; bands = interval-end step)

| N | model | band | R1 global affine | R2-R1 (+label-wise) | R3-R2 (+kNN nonlin) | unexplained | trace share | sym share | antisym share | glob abscissa | #expanding (of 16) | #complex pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | clf | 10-100 | 0.975 | +0.005 | +0.000 | 0.019 | 0.11 | 0.46 | 0.43 | +52.08e-3 | 15.3 | 4.9 |
| 250 | clf | 100-1000 | 0.972 | +0.003 | +0.000 | 0.024 | 0.24 | 0.40 | 0.35 | +0.76e-3 | 15.9 | 3.4 |
| 250 | clf | 1000+ | 0.976 | +0.003 | +0.000 | 0.021 | 0.23 | 0.41 | 0.37 | +0.07e-3 | 16.0 | 3.9 |
| 250 | AE | 10-100 | 0.967 | +0.008 | +0.001 | 0.023 | 0.03 | 0.51 | 0.45 | +80.27e-3 | 11.7 | 4.9 |
| 250 | AE | 100-1000 | 0.967 | +0.004 | +0.001 | 0.028 | 0.45 | 0.30 | 0.24 | +5.14e-3 | 16.0 | 3.9 |
| 250 | AE | 1000+ | 0.940 | +0.008 | +0.003 | 0.050 | 0.29 | 0.44 | 0.28 | +0.06e-3 | 12.6 | 4.7 |
| 1k | clf | 10-100 | 0.977 | +0.006 | +0.001 | 0.016 | 0.03 | 0.49 | 0.48 | +50.38e-3 | 13.0 | 5.6 |
| 1k | clf | 100-1000 | 0.972 | +0.005 | +0.000 | 0.022 | 0.14 | 0.45 | 0.42 | +1.64e-3 | 15.8 | 4.7 |
| 1k | clf | 1000+ | 0.980 | +0.003 | +0.000 | 0.018 | 0.25 | 0.40 | 0.35 | +0.08e-3 | 16.0 | 2.9 |
| 1k | AE | 10-100 | 0.969 | +0.006 | +0.002 | 0.023 | 0.03 | 0.52 | 0.45 | +77.99e-3 | 11.3 | 4.9 |
| 1k | AE | 100-1000 | 0.922 | +0.017 | +0.003 | 0.058 | 0.20 | 0.50 | 0.28 | +8.40e-3 | 14.9 | 4.3 |
| 1k | AE | 1000+ | 0.808 | +0.020 | +0.007 | 0.165 | 0.16 | 0.48 | 0.33 | +0.06e-3 | 11.4 | 5.5 |
| 4k | clf | 10-100 | 0.975 | +0.008 | +0.001 | 0.017 | 0.03 | 0.49 | 0.48 | +51.53e-3 | 12.9 | 5.6 |
| 4k | clf | 100-1000 | 0.907 | +0.019 | +0.000 | 0.073 | 0.00 | 0.50 | 0.50 | +4.29e-3 | 10.1 | 6.0 |
| 4k | clf | 1000+ | 0.978 | +0.004 | +0.000 | 0.018 | 0.24 | 0.40 | 0.37 | +0.07e-3 | 16.0 | 3.8 |
| 4k | AE | 10-100 | 0.969 | +0.008 | +0.001 | 0.022 | 0.02 | 0.52 | 0.46 | +77.37e-3 | 11.0 | 4.9 |
| 4k | AE | 100-1000 | 0.912 | +0.020 | +0.004 | 0.064 | 0.10 | 0.52 | 0.38 | +11.29e-3 | 12.3 | 4.7 |
| 4k | AE | 1000+ | 0.839 | +0.028 | +0.010 | 0.123 | 0.09 | 0.51 | 0.37 | +0.07e-3 | 7.1 | 5.7 |
| full | clf | 10-100 | 0.975 | +0.008 | +0.001 | 0.017 | 0.03 | 0.49 | 0.48 | +52.46e-3 | 12.5 | 5.6 |
| full | clf | 100-1000 | 0.860 | +0.036 | +0.001 | 0.103 | 0.00 | 0.51 | 0.49 | +5.76e-3 | 7.8 | 5.9 |
| full | clf | 1000+ | 0.577 | +0.073 | -0.000 | 0.350 | 0.00 | 0.51 | 0.49 | +0.52e-3 | 6.6 | 5.7 |
| full | AE | 10-100 | 0.969 | +0.006 | +0.002 | 0.022 | 0.02 | 0.52 | 0.46 | +74.39e-3 | 11.2 | 5.1 |
| full | AE | 100-1000 | 0.871 | +0.031 | +0.006 | 0.092 | 0.07 | 0.53 | 0.40 | +10.30e-3 | 11.8 | 4.9 |
| full | AE | 1000+ | 0.746 | +0.046 | +0.016 | 0.192 | 0.03 | 0.55 | 0.41 | +0.10e-3 | 7.3 | 5.9 |

### fashion_mnist (test probe, wd=0, mean over 3 seeds; bands = interval-end step)

| N | model | band | R1 global affine | R2-R1 (+label-wise) | R3-R2 (+kNN nonlin) | unexplained | trace share | sym share | antisym share | glob abscissa | #expanding (of 16) | #complex pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | clf | 10-100 | 0.984 | +0.004 | +0.000 | 0.011 | 0.05 | 0.49 | 0.46 | +56.34e-3 | 13.9 | 5.3 |
| 250 | clf | 100-1000 | 0.981 | +0.003 | +0.001 | 0.015 | 0.17 | 0.43 | 0.40 | +0.78e-3 | 16.0 | 4.9 |
| 250 | clf | 1000+ | 0.980 | +0.004 | +0.001 | 0.015 | 0.13 | 0.45 | 0.42 | +0.07e-3 | 15.4 | 4.5 |
| 250 | AE | 10-100 | 0.968 | +0.009 | +0.002 | 0.021 | 0.01 | 0.51 | 0.47 | +70.50e-3 | 10.1 | 5.5 |
| 250 | AE | 100-1000 | 0.961 | +0.011 | +0.002 | 0.026 | 0.43 | 0.32 | 0.25 | +5.04e-3 | 16.0 | 4.8 |
| 250 | AE | 1000+ | 0.935 | +0.016 | +0.004 | 0.045 | 0.36 | 0.38 | 0.27 | +0.07e-3 | 14.2 | 5.0 |
| 1k | clf | 10-100 | 0.985 | +0.005 | +0.001 | 0.009 | 0.03 | 0.49 | 0.48 | +48.90e-3 | 12.2 | 5.6 |
| 1k | clf | 100-1000 | 0.948 | +0.013 | +0.002 | 0.036 | 0.01 | 0.50 | 0.49 | +3.65e-3 | 12.7 | 5.5 |
| 1k | clf | 1000+ | 0.984 | +0.002 | +0.000 | 0.013 | 0.21 | 0.41 | 0.37 | +0.07e-3 | 15.7 | 4.4 |
| 1k | AE | 10-100 | 0.971 | +0.009 | +0.002 | 0.018 | 0.01 | 0.51 | 0.48 | +68.51e-3 | 9.9 | 5.7 |
| 1k | AE | 100-1000 | 0.884 | +0.038 | +0.006 | 0.072 | 0.22 | 0.47 | 0.32 | +6.85e-3 | 14.7 | 5.0 |
| 1k | AE | 1000+ | 0.809 | +0.046 | +0.010 | 0.134 | 0.19 | 0.46 | 0.34 | +0.08e-3 | 11.5 | 5.7 |
| 4k | clf | 10-100 | 0.985 | +0.005 | +0.001 | 0.009 | 0.02 | 0.50 | 0.48 | +46.57e-3 | 11.7 | 5.9 |
| 4k | clf | 100-1000 | 0.901 | +0.029 | +0.003 | 0.067 | 0.00 | 0.50 | 0.50 | +5.40e-3 | 8.6 | 5.7 |
| 4k | clf | 1000+ | 0.962 | +0.008 | +0.001 | 0.029 | 0.09 | 0.47 | 0.45 | +0.07e-3 | 15.2 | 5.0 |
| 4k | AE | 10-100 | 0.971 | +0.008 | +0.002 | 0.018 | 0.01 | 0.51 | 0.48 | +67.96e-3 | 10.1 | 5.7 |
| 4k | AE | 100-1000 | 0.866 | +0.047 | +0.008 | 0.079 | 0.10 | 0.52 | 0.38 | +7.04e-3 | 11.7 | 5.1 |
| 4k | AE | 1000+ | 0.790 | +0.061 | +0.010 | 0.139 | 0.08 | 0.51 | 0.41 | +0.09e-3 | 8.1 | 5.9 |
| full | clf | 10-100 | 0.985 | +0.005 | +0.001 | 0.009 | 0.03 | 0.49 | 0.48 | +51.16e-3 | 12.1 | 5.7 |
| full | clf | 100-1000 | 0.893 | +0.042 | +0.003 | 0.062 | 0.00 | 0.50 | 0.49 | +6.49e-3 | 7.3 | 5.8 |
| full | clf | 1000+ | 0.601 | +0.079 | +0.006 | 0.314 | 0.00 | 0.51 | 0.48 | +0.53e-3 | 6.7 | 5.6 |
| full | AE | 10-100 | 0.969 | +0.008 | +0.003 | 0.019 | 0.01 | 0.51 | 0.48 | +63.91e-3 | 9.7 | 5.5 |
| full | AE | 100-1000 | 0.839 | +0.058 | +0.009 | 0.094 | 0.07 | 0.53 | 0.39 | +7.98e-3 | 10.9 | 5.3 |
| full | AE | 1000+ | 0.756 | +0.072 | +0.012 | 0.160 | 0.02 | 0.55 | 0.42 | +0.10e-3 | 7.2 | 5.8 |

## Further tables (A linear AE, B labels vs k-means, C train vs test, D gap, E dimension)

### A. linear AE (d=16, wd=0), test probe: R1 / unexplained by band

| dataset | N | R1 10-100 | R1 100-1000 | R1 1000+ | unexpl 1000+ | R2-R1 1000+ | trace/sym/anti share 1000+ |
|---|---|---|---|---|---|---|---|
| mnist | 250 | 0.906 | 0.812 | 0.745 | 0.182 | +0.064 | 0.01/0.66/0.33 |
| mnist | 1k | 0.913 | 0.849 | 0.663 | 0.248 | +0.075 | 0.06/0.60/0.33 |
| mnist | 4k | 0.900 | 0.821 | 0.578 | 0.313 | +0.094 | 0.15/0.52/0.26 |
| mnist | full | 0.897 | 0.800 | 0.386 | 0.446 | +0.143 | 0.18/0.47/0.24 |
| fashion_mnist | 250 | 0.935 | 0.864 | 0.834 | 0.121 | +0.041 | 0.06/0.67/0.24 |
| fashion_mnist | 1k | 0.943 | 0.912 | 0.828 | 0.134 | +0.034 | 0.04/0.68/0.23 |
| fashion_mnist | 4k | 0.936 | 0.897 | 0.858 | 0.110 | +0.030 | 0.10/0.67/0.21 |
| fashion_mnist | full | 0.928 | 0.871 | 0.805 | 0.149 | +0.043 | 0.14/0.64/0.20 |

### B. label-wise vs label-free k-means(10)-wise affine increment R2-R1 (AE MLP d16; all N, wd, seeds pooled; paired by run)

| dataset | probe | band | R2-R1 labels | R2-R1 k-means | labels - kmeans (paired mean +- se) | frac runs labels>kmeans |
|---|---|---|---|---|---|---|
| mnist | test | 10-100 | +0.0067 | +0.0077 | -0.0010 +- 0.0003 | 0.29 |
| mnist | test | 100-1000 | +0.0175 | +0.0172 | +0.0003 +- 0.0003 | 0.42 |
| mnist | test | 1000+ | +0.0300 | +0.0324 | -0.0024 +- 0.0005 | 0.17 |
| mnist | train | 10-100 | +0.0053 | +0.0062 | -0.0009 +- 0.0003 | 0.33 |
| mnist | train | 100-1000 | +0.0169 | +0.0162 | +0.0006 +- 0.0005 | 0.67 |
| mnist | train | 1000+ | +0.0285 | +0.0296 | -0.0011 +- 0.0004 | 0.25 |
| fashion_mnist | test | 10-100 | +0.0080 | +0.0146 | -0.0067 +- 0.0003 | 0.00 |
| fashion_mnist | test | 100-1000 | +0.0295 | +0.0380 | -0.0085 +- 0.0013 | 0.00 |
| fashion_mnist | test | 1000+ | +0.0392 | +0.0463 | -0.0071 +- 0.0009 | 0.00 |
| fashion_mnist | train | 10-100 | +0.0021 | +0.0126 | -0.0105 +- 0.0019 | 0.04 |
| fashion_mnist | train | 100-1000 | +0.0277 | +0.0378 | -0.0100 +- 0.0017 | 0.08 |
| fashion_mnist | train | 1000+ | +0.0383 | +0.0443 | -0.0060 +- 0.0010 | 0.08 |

### C. train probe minus test probe (AE MLP d16, paired by run; N>=1000 where both probes have 1000 points)

| dataset | band | metric | train | test | train - test (+- se) | n runs |
|---|---|---|---|---|---|---|
| mnist | 10-100 | R1 | 0.970 | 0.972 | -0.0012 +- 0.0003 | 18 |
| mnist | 10-100 | R2-R1 (labels) | 0.006 | 0.006 | -0.0003 +- 0.0002 | 18 |
| mnist | 10-100 | R3-R2 | 0.002 | 0.001 | +0.0003 +- 0.0002 | 18 |
| mnist | 10-100 | unexplained | 0.022 | 0.021 | +0.0013 +- 0.0003 | 18 |
| mnist | 10-100 | trace share | 0.019 | 0.019 | +0.0007 +- 0.0004 | 18 |
| mnist | 10-100 | antisym share | 0.461 | 0.461 | +0.0001 +- 0.0007 | 18 |
| mnist | 100-1000 | R1 | 0.900 | 0.906 | -0.0061 +- 0.0007 | 18 |
| mnist | 100-1000 | R2-R1 (labels) | 0.022 | 0.022 | +0.0006 +- 0.0008 | 18 |
| mnist | 100-1000 | R3-R2 | 0.005 | 0.004 | +0.0009 +- 0.0002 | 18 |
| mnist | 100-1000 | unexplained | 0.072 | 0.068 | +0.0047 +- 0.0006 | 18 |
| mnist | 100-1000 | trace share | 0.096 | 0.095 | +0.0011 +- 0.0015 | 18 |
| mnist | 100-1000 | antisym share | 0.386 | 0.388 | -0.0023 +- 0.0019 | 18 |
| mnist | 1000+ | R1 | 0.759 | 0.771 | -0.0118 +- 0.0013 | 18 |
| mnist | 1000+ | R2-R1 (labels) | 0.038 | 0.037 | +0.0005 +- 0.0007 | 18 |
| mnist | 1000+ | R3-R2 | 0.015 | 0.013 | +0.0020 +- 0.0004 | 18 |
| mnist | 1000+ | unexplained | 0.188 | 0.179 | +0.0092 +- 0.0012 | 18 |
| mnist | 1000+ | trace share | 0.129 | 0.120 | +0.0083 +- 0.0029 | 18 |
| mnist | 1000+ | antisym share | 0.356 | 0.362 | -0.0059 +- 0.0022 | 18 |
| fashion_mnist | 10-100 | R1 | 0.970 | 0.971 | -0.0011 +- 0.0002 | 18 |
| fashion_mnist | 10-100 | R2-R1 (labels) | 0.006 | 0.008 | -0.0020 +- 0.0006 | 18 |
| fashion_mnist | 10-100 | R3-R2 | 0.002 | 0.003 | -0.0002 +- 0.0003 | 18 |
| fashion_mnist | 10-100 | unexplained | 0.021 | 0.018 | +0.0033 +- 0.0005 | 18 |
| fashion_mnist | 10-100 | trace share | 0.007 | 0.007 | +0.0002 +- 0.0002 | 18 |
| fashion_mnist | 10-100 | antisym share | 0.483 | 0.482 | +0.0009 +- 0.0005 | 18 |
| fashion_mnist | 100-1000 | R1 | 0.885 | 0.893 | -0.0077 +- 0.0017 | 18 |
| fashion_mnist | 100-1000 | R2-R1 (labels) | 0.036 | 0.037 | -0.0003 +- 0.0014 | 18 |
| fashion_mnist | 100-1000 | R3-R2 | 0.007 | 0.006 | +0.0007 +- 0.0003 | 18 |
| fashion_mnist | 100-1000 | unexplained | 0.072 | 0.064 | +0.0073 +- 0.0012 | 18 |
| fashion_mnist | 100-1000 | trace share | 0.101 | 0.096 | +0.0058 +- 0.0019 | 18 |
| fashion_mnist | 100-1000 | antisym share | 0.395 | 0.399 | -0.0041 +- 0.0020 | 18 |
| fashion_mnist | 1000+ | R1 | 0.810 | 0.827 | -0.0170 +- 0.0034 | 18 |
| fashion_mnist | 1000+ | R2-R1 (labels) | 0.050 | 0.048 | +0.0013 +- 0.0009 | 18 |
| fashion_mnist | 1000+ | R3-R2 | 0.010 | 0.009 | +0.0015 +- 0.0004 | 18 |
| fashion_mnist | 1000+ | unexplained | 0.130 | 0.116 | +0.0143 +- 0.0035 | 18 |
| fashion_mnist | 1000+ | trace share | 0.130 | 0.118 | +0.0125 +- 0.0044 | 18 |
| fashion_mnist | 1000+ | antisym share | 0.384 | 0.391 | -0.0074 +- 0.0028 | 18 |

#### classifier FM (phase grid, wd 0 and 1e-4, N>=1000): train - test

| band | metric | train | test | diff |
|---|---|---|---|---|
| 10-100 | R1 | 0.984 | 0.985 | -0.0006 |
| 10-100 | R2-R1 | 0.005 | 0.005 | -0.0003 |
| 10-100 | unexplained | 0.010 | 0.009 | +0.0010 |
| 100-1000 | R1 | 0.906 | 0.909 | -0.0029 |
| 100-1000 | R2-R1 | 0.031 | 0.029 | +0.0027 |
| 100-1000 | unexplained | 0.060 | 0.059 | +0.0010 |
| 1000+ | R1 | 0.757 | 0.763 | -0.0062 |
| 1000+ | R2-R1 | 0.054 | 0.044 | +0.0103 |
| 1000+ | unexplained | 0.187 | 0.189 | -0.0024 |

### D. within-(N, wd) Spearman of late-band (1000+) quantity vs final recon gap (test-train), AE MLP d16; group-demeaned, permutation p (within groups)

| dataset | probe | quantity | rho | p | n |
|---|---|---|---|---|---|
| mnist | test | R1 | -0.04 | 0.864 | 24 |
| mnist | test | dR2 | -0.17 | 0.435 | 24 |
| mnist | test | dR3 | +0.05 | 0.853 | 24 |
| mnist | test | U | +0.11 | 0.676 | 24 |
| mnist | test | trsh | +0.20 | 0.479 | 24 |
| mnist | test | antish | -0.38 | 0.182 | 24 |
| mnist | test | iso | -0.25 | 0.364 | 24 |
| mnist | test | Lnorm | -0.38 | 0.187 | 24 |
| mnist | train | R1 | -0.03 | 0.904 | 24 |
| mnist | train | dR2 | -0.08 | 0.707 | 24 |
| mnist | train | dR3 | +0.27 | 0.202 | 24 |
| mnist | train | U | +0.09 | 0.700 | 24 |
| mnist | train | trsh | +0.11 | 0.702 | 24 |
| mnist | train | antish | -0.27 | 0.348 | 24 |
| mnist | train | iso | -0.25 | 0.342 | 24 |
| mnist | train | Lnorm | -0.13 | 0.657 | 24 |
| fashion_mnist | test | R1 | -0.20 | 0.382 | 24 |
| fashion_mnist | test | dR2 | +0.37 | 0.093 | 24 |
| fashion_mnist | test | dR3 | +0.06 | 0.797 | 24 |
| fashion_mnist | test | U | +0.03 | 0.923 | 24 |
| fashion_mnist | test | trsh | -0.55 | 0.008 | 24 |
| fashion_mnist | test | antish | +0.44 | 0.036 | 24 |
| fashion_mnist | test | iso | -0.29 | 0.203 | 24 |
| fashion_mnist | test | Lnorm | +0.17 | 0.480 | 24 |
| fashion_mnist | train | R1 | -0.25 | 0.306 | 24 |
| fashion_mnist | train | dR2 | +0.17 | 0.392 | 24 |
| fashion_mnist | train | dR3 | +0.18 | 0.429 | 24 |
| fashion_mnist | train | U | +0.21 | 0.435 | 24 |
| fashion_mnist | train | trsh | -0.41 | 0.065 | 24 |
| fashion_mnist | train | antish | +0.45 | 0.046 | 24 |
| fashion_mnist | train | iso | -0.26 | 0.264 | 24 |
| fashion_mnist | train | Lnorm | +0.16 | 0.465 | 24 |
| both | test | R1 | -0.14 | 0.412 | 48 |
| both | test | dR2 | +0.09 | 0.560 | 48 |
| both | test | dR3 | +0.05 | 0.736 | 48 |
| both | test | U | +0.05 | 0.748 | 48 |
| both | test | trsh | -0.17 | 0.313 | 48 |
| both | test | antish | +0.06 | 0.702 | 48 |
| both | test | iso | -0.27 | 0.124 | 48 |
| both | test | Lnorm | -0.07 | 0.686 | 48 |
| both | train | R1 | -0.16 | 0.361 | 48 |
| both | train | dR2 | +0.02 | 0.879 | 48 |
| both | train | dR3 | +0.22 | 0.141 | 48 |
| both | train | U | +0.13 | 0.461 | 48 |
| both | train | trsh | -0.12 | 0.460 | 48 |
| both | train | antish | +0.11 | 0.529 | 48 |
| both | train | iso | -0.24 | 0.163 | 48 |
| both | train | Lnorm | +0.02 | 0.924 | 48 |

Between-group (N confounded) Spearman for contrast, test probe, both datasets:

| quantity | rho(gap_final) | rho(log N) |
|---|---|---|
| R1 | +0.70 | -0.78 |
| dR2 | -0.84 | +0.83 |
| U | -0.60 | +0.74 |
| trsh | +0.33 | -0.30 |

### E. latent dimension (AE MLP, wd 0, test probe, N in {1k, full}): R1 / unexplained late

| dataset | d | N | R1 100-1000 | R1 1000+ | unexpl 1000+ | R2-R1 1000+ | #expanding/d |
|---|---|---|---|---|---|---|---|
| mnist | 8 | 1k | 0.882 | 0.731 | 0.209 | +0.043 | 6.8/8 |
| mnist | 8 | full | 0.711 | 0.512 | 0.348 | +0.104 | 3.5/8 |
| mnist | 16 | 1k | 0.922 | 0.808 | 0.165 | +0.020 | 11.4/16 |
| mnist | 16 | full | 0.871 | 0.746 | 0.192 | +0.046 | 7.3/16 |
| mnist | 64 | 1k | 0.943 | 0.879 | 0.113 | +0.005 | 29.5/64 |
| mnist | 64 | full | 0.942 | 0.907 | 0.080 | +0.008 | 23.7/64 |
| fashion_mnist | 8 | 1k | 0.863 | 0.765 | 0.144 | +0.077 | 6.6/8 |
| fashion_mnist | 8 | full | 0.774 | 0.607 | 0.240 | +0.126 | 3.2/8 |
| fashion_mnist | 16 | 1k | 0.884 | 0.809 | 0.134 | +0.046 | 11.5/16 |
| fashion_mnist | 16 | full | 0.839 | 0.756 | 0.160 | +0.072 | 7.2/16 |
| fashion_mnist | 64 | 1k | 0.909 | 0.849 | 0.133 | +0.011 | 32.1/64 |
| fashion_mnist | 64 | full | 0.916 | 0.880 | 0.098 | +0.017 | 29.8/64 |
## Figures
- `lds_fig1_R1_time.png`: R1 (solid) and R2 (dashed) vs step; classifier vs AE-MLP vs linear AE, per dataset and N.
- `lds_fig2_generator_shares.png`: trace / symmetric / antisymmetric shares of the generator by band (N=1k).
- `lds_fig3_labels_traintest_gap.png`: label vs k-means class-wise increment; train vs test probe unexplained; unexplained vs recon gap (colour = N).

## Caveats
- **Gauge.** Latent coordinates are not pinned between snapshots (AE decoder pins them only up to invertible reparametrization drift; the classifier probe is passive). A_t absorbs time-dependent gauge; eigenvalues/rotation-vs-shear shares are only approximately gauge free. R1 uses the raw frame, uncentered (translation counts as affine), so R1 is partly inflated by global drift/scaling; sym ~ antisym ~ 0.5 is also what a generic matrix gives. We did not compute a gauge-null.
- **Light partition mode.** For CPU feasibility the AE runs use `light=True` (kernel ridge with a reduced grid; no joint kernel fits, no kernel-shuffle null). R3 is therefore the kNN(10) variant; the light R3_krr (stored as `R3_krr`) was sometimes below R2 and is not used. The classifier R3 numbers come from the full run (the kNN column is used there as well). Hence R3-R2 is a lower bound on what a richer nonlinear fit would get; "unexplained" is an upper bound. No train-time or shuffled-sample null was run for the AE.
- **Matching.** Classifier "full" = 60000, AE "full" = 60000; classifier uses its phase grid (wd 0 shown; wd 1e-4 exists but not tabulated), classifier MNIST has no train-probe phase runs, so train-vs-test is shown for AE (both datasets) and classifier FM only. Classifier FM generator rows were recomputed here from latents (same function) because the FM partition files do not store them.
- **Probe size.** Train probes for N=250 have 250 points (125/125 splits for a 16x17-parameter affine fit), so N=250 train-probe numbers are not comparable and were excluded from the train-vs-test table.
- **Steps >= 1000 band mixes** the post-fit-onset regime; the AE reconstruction is still improving there while the classifier is at its terminal phase, so "late" is not the same training regime. Seeds: 3 per cell; se of paired differences shown, no multiple-comparison correction beyond saying so.
- **k-means groups** are fixed 50-d pixel-PCA k-means(10) fitted on the full test split (as in `ae_analysis.py`), not on latents; label-free in the sense of not using y.
- Mechanisms (why the linear AE is not exactly affine, why the AE keeps losing affinity at N=1k) are untested hypotheses. No novelty claims.
