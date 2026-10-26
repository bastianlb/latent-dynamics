# latent-dynamics — project spec & data contract

Study how the **latent representations** of a small neural model evolve **over
training iterations**: track a fixed probe set of samples, snapshot their
latents on a schedule, then compute geometric and topological metrics over the
trajectory.

## Paths (storage policy — see CLAUDE.md)

- **Code** lives in `/homes/lbastian/latent-dynamics/` (this repo) as the installable package `latentdyn` (`src/latentdyn/`; see README.md).
- **All bulky artifacts** live under
  `ARTIFACTS = /vol/bitbucket/lbastian/latent-dynamics/`.
  Never write large/growing files to `/data` or `/homes`.

Layout under `ARTIFACTS`:

```
data/                         # torchvision MNIST cache (download_root)
runs/<run_id>/
  config.json                 # full resolved config (see below)
  probe_index.npz             # fixed probe-set sample indices
  latents/step_{step:07d}.npz # one latent snapshot per logging step (fixed TEST probe set)
  latents_train/step_{step:07d}.npz # same steps/keys, fixed TRAIN probe set (only with --probe_train)
  probe_train_index.npz       # train probe: indices (into full train split), labels (true), [noisy_labels]
  train_index.npz             # only with --train_subset/--label_noise: pool indices, labels, [noisy_labels, noisy_mask]
  model_final.pt              # final weights (single file, not per-step)
  metrics/timeseries.npz      # computed metrics over time (analysis output)
  metrics/timeseries.csv      # same, human-readable
  figures/*.png               # plots
```

`run_id` format: `mnist_probe_d{latent_dim}_{tag}` (tag optional). Keep it
deterministic/overridable via config — no timestamps that break resume.

## Data contract — latent snapshot (`latents/step_{step:07d}.npz`)

Each `.npz` contains:

- `latents`  : float32, shape `[N, D]`  — encoder output for the probe set
- `labels`   : int64,   shape `[N]`     — class label per probe sample (0..9)
- `step`     : int64 scalar             — global optimizer step
- `epoch`    : float32 scalar           — fractional epoch at this step
- `train_loss` : float32 scalar         — training loss at this step
- `train_acc`  : float32 scalar         — running/train accuracy at this step
- `test_acc`   : float32 scalar         — test accuracy at this step (probe or full test set; document which)

Added in v2 (extra keys; old loaders ignore them, old runs simply lack them):

- `test_loss` : float32 scalar — full-test-set CE loss at this step (eval mode)
- `lr` : float32 scalar — optimizer learning rate at this step
- `train_loss_running`, `train_acc_running` : float32 — mean over the training batches since the previous
  snapshot (train mode, augmented/noisy-labelled batches as seen by the optimizer; NaN at step 0)
- `l2_scale` : float32 scalar — current head scale; present only when `--l2_scale` is set. `latents` are
  always the unit-norm z (PRE-scale); the head sees `l2_scale * z`.
- `noisy_labels` : int64 `[N]` — **only in `latents_train/`, only with `--label_noise`**. There `labels` are the
  TRUE labels and `noisy_labels` the labels actually trained on.

`train_loss` / `train_acc` semantics: with `--train_eval running` (default) = the running mean above
(step 0: eval-mode full pass over the train pool); with `--train_eval full` = eval-mode full pass over
the whole un-augmented train pool (noisy labels if `--label_noise`) at every snapshot. `test_acc`/`test_loss`
are always the FULL test set. `latents_train/step_*.npz` carries the identical scalars as `latents/step_*.npz`
and the same keys (+`noisy_labels`), so analysis can be pointed at either directory unchanged.

Train probe set: `n_probe` examples, class-balanced (unless `--no_probe_balanced`), drawn from the training
POOL (the `--train_subset` subset if given), fixed per seed (`RandomState(seed+3000)`), un-augmented.
Training pool: `--train_subset N` = class-balanced subset (`RandomState(seed+1000)`); `--label_noise p` =
exactly round(p*|pool|) labels replaced by a uniform random wrong label (`RandomState(seed+2000)`); test labels
untouched.

Snapshot schedule: `--log_schedule linear` (default) = step 0, every `log_every`, `dense_early`, final.
`--log_schedule log` = step 0, steps 1..10, then geometric spacing to the final step, ~`n_snapshots` (80)
distinct integer steps (final step always included).

Snapshots are taken in eval mode under `no_grad`, fp32 (never autocast); BatchNorm running stats are not
touched and `train()` mode is restored. The only coupling to training is the global torch RNG (DataLoader
iterator creation); `--isolate_snapshot_rng` restores the RNG state after each snapshot so the trained
model is bit-identical for any snapshot schedule / probe settings.

Disk per run (float32, uncompressed npz): one snapshot = N*D*4 B per probe set. ResNet-18, D=512, N=1000, 80
snapshots, both probe sets: 2*1000*512*4 B = ~4.1 MB/snapshot -> ~330 MB (+45 MB model_final); with N=2000 ~660 MB.
MNIST MLP D=16, N=1000, 80 snapshots, both sets: ~10 MB.

The probe set is **fixed for the whole run** (same N samples, same order, in
every snapshot) so latents are directly comparable across steps. Indices are
saved once in `probe_index.npz` (`indices` int64 `[N]`, plus `labels`).

## config.json (resolved config)

Include at least: `dataset`, `latent_dim` (D), `n_probe` (N), `probe_balanced`
(bool), `log_every` (steps), `epochs`, `batch_size`, `lr`, `seed`,
`encoder_arch`, `recon` (bool, whether a decoder/reconstruction term is used),
`device`, `run_id`, `artifacts_root`.

## Metrics timeseries (analysis output — `metrics/timeseries.npz`/`.csv`)

One row per snapshot step (sorted by `step`). Columns = `step`, `epoch`,
`train_loss`, `train_acc`, `test_acc`, plus one column per scalar metric defined
in `src/latentdyn/metrics.py`. Array-valued metrics (e.g. per-class-pair distances) go in the
`.npz` under their own keys, indexed consistently with `step`.

## Conventions

- Seed everything; make runs reproducible.
- Prefer compact dtypes; do not save full-dataset latents or per-step dense
  checkpoints.
- GPU launches are handled by the coordinator per CLAUDE.md — library code must
  respect `CUDA_VISIBLE_DEVICES` and a configurable `device`, and must **not**
  hard-pin or grab all GPUs.

## Optimizer flags and diagnostics (`--adam_eps`, `--adam_beta2`, `--log_opt_diag`)

- `--adam_eps` (default 1e-8) and `--adam_beta2` (default 0.999) are passed to `torch.optim.Adam` (beta1 stays 0.9).
  Defaults reproduce the previous behaviour exactly. Both are stored in `config.json`.
- **Weight-decay convention.** `--weight_decay` is a *coupled* L2 penalty (added to the gradient inside Adam/SGD; this is
  not AdamW). It is equivalent to minimising `L(theta) + 0.5*wd*||theta||^2` over all parameters except the learnable
  `l2_scale`; the penalty's Hessian is `wd*I`. All diagnostics below use this full loss.
- `--log_opt_diag` (default off) adds the following keys to every snapshot npz (both `latents/` and `latents_train/`,
  identical values). Code: `latentdyn/train_aux/optdiag.py`. The "fixed batch" is `min(1024, |train pool|)` examples drawn once without
  replacement from the un-augmented training pool (noisy labels if any) with `np.random.RandomState(seed+5000)`;
  the loss is evaluated in eval mode, fp32. The power-iteration start vector uses a dedicated
  `torch.Generator` (seed+7000). Diagnostics use only `torch.autograd.grad` (no `.grad` is written), only read the optimizer
  state, and `train()` mode is restored, so the training trajectory is bit-identical with and without the flag
  (checked, Adam and SGD).

| key | shape/dtype | meaning |
|---|---|---|
| `opt_theta_norm` | () f32 | `||theta||` over all parameters |
| `opt_dtheta_norm` | () f32 | `||theta_t - theta_{t-1}||` of the last optimizer step (NaN at step 0) |
| `opt_grad_norm` | () f32 | `||g||` of the full loss (incl. wd term) on the fixed batch |
| `opt_lambda_max` | () f32 | top-magnitude Hessian eigenvalue of the full loss on the fixed batch; 20 power iterations with Hessian-vector products (double backward), warm-started from the previous snapshot's vector |
| `opt_lambda_iters_resid` | () f32 | `||Hv - lambda v|| / |lambda|` at the last iteration (convergence residual) |
| `opt_eff_step_q` (Adam) | (5,) f32 | quantiles 1,10,50,90,99 % over all parameters of `lr/(sqrt(v_hat)+eps)`, `v_hat = exp_avg_sq/(1-beta2^t)` (NaN at step 0) |
| `opt_vhat_mean` (Adam) | () f32 | mean of `v_hat` (NaN at step 0) |
| `opt_lambda_max_precond` (Adam) | () f32 | top eigenvalue of `P^{1/2} H P^{1/2}`, `P = diag(1/(sqrt(v_hat)+eps))` (lr not included: lr*value is the stability-relevant number), same 20-iter warm-started power method (NaN at step 0) |
| `opt_lambda_max_precond_resid` (Adam) | () f32 | its convergence residual |

Adam-only keys are absent for SGD. Power iteration finds the largest-|eigenvalue|; late in training this is the top
positive curvature. Heavy-ball SGD (momentum beta) is linearly stable for `lr*lambda_max < 2(1+beta)` (=3.8 at beta=0.9).
Cost: about 0.5 s per snapshot on CPU (2 threads) for the MLP.

## Dense optimizer-state logging (`--log_opt_dense`)

Adam only; default off (training bit-identical on/off: reads optimizer state and `p.grad` only, no RNG, no extra
backward). Code: `latentdyn/train_aux/optdense.py`, hooked in `latentdyn/train.py` right after `optimizer.step()` (before the next `zero_grad`).
Definitions at step `t` (state step count): `vhat = exp_avg_sq/(1-b2^t)`, `mhat = exp_avg/(1-b1^t)`,
`g' = p.grad + wd*theta_before` (coupled L2, wd=0 for `l2_scale`), `denom = sqrt(vhat)+eps`, `Dtheta = lr*mhat/denom`,
`a = lr/denom`, `r = |mhat|/denom`. Scopes: `global`, `fc1` (`encoder.net.1`), `fc2` (`encoder.net.3`),
`fc3` (`encoder.net.5`), `probe` (`probe.*`). Quantiles = nearest rank on a sorted copy.

Flags (all in `latentdyn/config.py`; every default is off/neutral):

| flag | default | meaning |
|---|---|---|
| `--log_opt_dense` | off | enable |
| `--opt_dump_dir` | `/data2/lbastian/latent-dynamics/opt_dense/<run_id>` | bulky dumps (b),(c). Scalars/snapshots go to `--artifacts_root` |
| `--opt_ref_onset` | none | onset step: (b) near-window `[ref-2000, ref+150]`; default fixed window `[ref-500, ref+150]` |
| `--opt_dump_every` / `--opt_dump_near_every` | 200 / 20 | (b) stride overall / in the near window |
| `--opt_arm_start`, `--opt_arm_end` | none | ring-buffer (event) mode: armed region. Setting `--opt_arm_start` selects ring mode |
| `--opt_ring_steps`, `--opt_post_steps` | 400, 150 | ring length before the trigger step, steps recorded after it |
| `--opt_fixed_start`, `--opt_fixed_end`, `--opt_fixed_every` | none, none, 5 | fixed-window (non-event) mode |
| `--opt_quant_every` | 10 | quantile columns every N steps outside armed windows (every step inside; NaN otherwise) |
| `--opt_cap_gb` | 1.5 | per-run cap on bulky dumps |
| `--opt_min_free_gb`, `--opt_max_total_gb` | 50, 40 | disk guard: if free space on the dump drive is below the first, or the parent `opt_dense/` already exceeds the second, bulky dumps are refused with a warning (training continues; the scalar file is still written) |
| `--opt_scalar_save_every` | 5000 | periodic save of `dense_scalars.npy` (also on finish and on SIGTERM) |

Event mode (ring): inside `[arm_start, arm_end]` the last `ring_steps+1` steps of (vhat, mhat, g') are kept in RAM as
bfloat16 (`ring_steps+1` entries: steps `t*-400..t*`). Trigger: minibatch loss >= 3x median of the previous 50 minibatch
losses AND `||g'||` >= 5x the median of the previous 50 `||g'||` (history is kept for all steps, needs 50 entries). On the first
trigger `t*` the ring is flushed and `post_steps` further steps are recorded (551 steps total); one flush per run. If the
armed region ends with no trigger, the last ring is flushed and `no_trigger=True`.
Non-event mode (fixed): every `opt_fixed_every` steps in the fixed window, vectors recorded and written at the window end.

Outputs:
- `<run_dir>/dense_scalars.npy` float32 `[total_steps, ncols]` (row `t-1` = step `t`), column names in
  `<run_dir>/dense_scalars_cols.json`: globals `loss, gprime_norm, grad_norm, theta_norm, dtheta_norm, step, lr`; then per scope
  `S/vhat_q{1e-4,0.001,0.01,0.1,0.5}`, `S/r_q{1e-4,0.001,0.01,0.1,0.5,0.9,0.99,0.999,0.9999}`, `S/frac_sqrtv_lt_eps`,
  `S/frac_sqrtv_lt_10eps`, `S/max_a`, `S/max_dtheta_over_lr`, `S/frac_dtheta_gt_lr`, `S/frac_dtheta_gt_2lr`, `S/dtheta_norm`
  (112 columns). `<run_dir>/dense_meta.json`: mode, trigger step, bytes written, guard messages.
- `<dump_dir>/vhat_bf16.npz`: `steps` int64 `[n]`, `vhat_u16` uint16 `[n, P]` (bfloat16 bit patterns, P=218618; decode with
  `(u16.astype(np.uint32) << 16).view(np.float32)`).
- `<dump_dir>/c_dump.npz`: `steps`, `vhat_u16`, `mhat_u16`, `gp_u16` (g' = coupled gradient), all uint16 bf16 bits `[n, P]`;
  `trig_step` (-1 if none), `no_trigger` bool, `mode` ("ring"/"fixed"), `P`.
- Not implemented: the 128 minibatch indices per dumped step (the DataLoader does not expose them without changing the
  loader). bfloat16 conversion is round-to-nearest-even via `Tensor.to(torch.bfloat16)`.
