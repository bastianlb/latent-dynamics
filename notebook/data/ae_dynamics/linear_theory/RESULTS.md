# Linear AE timescale law: lambda^-1 vs lambda^-2

Script: /homes/lbastian/latent-dynamics/linear_ae_theory.py (+ linear_ae_theory_plot.py). Figure: linear_ae_theory.png. Raw: summary.json, traj_*.npz, run.log.
Setup: MNIST normalised pixels, fixed 10k train subset, d=16 linear AE, default nn.Linear init or 1e-3x. Adam lr1e-3 bs128 (with biases, 30k steps);
SGD = exact full-batch gradient descent on centred data via the covariance (no biases), lr=0.5*D/(16*lam0)=0.448 (default dec init has col-norm^2~16, so the
sharpness at init is 16x larger than at the solution; lr 2.9 diverged), 10k steps. 3 seeds. Mode strength s_k = v_k^T Wd We v_k (v_k = PCA eigvectors);
rise = step at which s_k is halfway s(0)->final (log-interp), top 16 modes. R2 rise = ae_analysis.affine_readout_r2 + rise_steps on the encoder code (3000 pts).

Fitted slope of log(rise step) vs log(lambda_k), mean +- sd over 3 seeds:

| condition | strength s_k | affine-readout R2 | theory (SGD ODE) |
|---|---|---|---|
| SGD default init | -1.16 +- 0.01 | -1.59 +- 0.20 | -1.00 |
| SGD small init (1e-3) | -1.07 +- 0.03 | -2.55 +- 0.14 | -1.00 |
| Adam default init | -1.17 +- 0.01 | -1.59 +- 0.07 | n/a |
| Adam small init | -1.36 +- 0.06 | -4.61 +- 0.22 | n/a |

Conclusions
- The decoder/encoder mode strength follows ~lambda^-1.1 to -1.4 in every condition, including Adam with default init (the actual setting): -1.17 +- 0.01.
  So the singular-value rise is close to the theory law. Adam alone shifts it little at default init (-1.17 vs -1.16); at small init Adam steepens it to -1.36.
- The steeper law (-1.6 here, -1.9/-2.0 in the real runs) is mainly (c), a measurement artefact: the affine-readout R2 is scale-free. It rises when the encoder
  direction aligns with PC k, not when its strength reaches its final value. With small init the R2 "rise" occurs within 1-10 steps (step 1 for the strongest PCs),
  long before the strengths move, and gives slopes -2.5 (SGD) and -4.6 (Adam). Because the 50% criterion is baseline-subtracted from a random-init readout,
  the number depends on init scale, not only on dynamics.
- (b) default init matters through the readout baseline (random init already reads out the top PCs) and a mild slope change (-1.16 vs -1.07), not through the strength law.
- Caveats: our R2 slope here (-1.59) is shallower than in the real runs (-1.9..-2.0; full N, other data order, wd0); that residual gap is not reproduced by this
  10k-subset setup and was not chased. The theory overlay uses balanced logistic/ODE modes; for SGD small init it sits ~0.7x below the data in absolute time (slope
  -1.00 vs -1.07), but for default init the ODE (delta from init col-norms) is off by ~100x in absolute time (slope still -1), so default-init theory is slope-only.
  Slowest mode (16) in SGD default has not converged by 10k steps (final s 0.5-0.9), so its rise time is biased. Modes with close lambda (15,16) interact.
- Staged PCA learning (order and ~1/lambda strength rise) is a sound anchor for the singular-value dynamics of the linear AE; R2-readout rise times should not be
  compared quantitatively to t ~ 1/lambda.
