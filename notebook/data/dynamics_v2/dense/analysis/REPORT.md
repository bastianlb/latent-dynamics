# Dense optimizer-state block: findings (2026-10-16)

The analysis was done by a subagent with `dense_analysis.py`, and the orchestrator transcribed this report. There are 12 single-seed runs, and all numbers are descriptive. The key figure is `overview_loss_fraclow.png`.

## 0. Sanity checks and a correction
- All 8 ring triggers fall in (previous snapshot, snapshot onset]. In 7 of 8 runs the loss reaches 10× its recent median within 0–18 steps of the trigger.
- Run #3 triggered on a precursor 658 steps early. Its main spike, at step 36323, falls outside the dumped window.
- Runs #4 and #5 never spike: their maximum loss ratios are 6.0 and 3.7.
- **Every wd=1e-4 run spikes repeatedly at the step level.** Each has 4–8 large spikes per 40k steps, at near-regular intervals: about 4–5.8k steps at N=4000 and 7–11k steps at N=1000. Each spike lasts about 85 steps.
- Snapshots caught only 9 of 33 episodes (27%). The snapshot-level type-F event fractions therefore measure snapshot-sampling chance, not whether a run spikes.
- Runs #9 and #10, which were labelled "none" from snapshots, actually spike. Pairs D and E are therefore not boundary pairs.

## 1. v̂ collapse
- **Type N (wd=0, eps ≥ 1e-7):** frac(√v̂ < eps) is about 1 at the trigger, and median √v̂/eps is 0.02–0.035.
- The non-spikers #4 and #5 are in the same state. The collapse is necessary but not sufficient.
- **Type F at eps ≤ 1e-7:** frac(√v̂ < eps) is about 0, with median √v̂/eps between 200 and 8200. The eps-floor mechanism cannot explain type F.

## 2. Which parameters move
- The update is diffuse. Half of the update energy needs 23–46% of all coordinates.
- Layer shares are proportional to layer size, and 84–119 of the 256 fc1 hidden units carry half of the energy. There is no localized set of units.

## 3. Temporal order
- The trigger reaches the loss peak in 2–23 steps.
- For type N, the v̂ collapse comes 3.8k–28k steps before the spike.
- Type N shows slow growth in ‖g‖ before the spike, ×1.5–2.3 over about 400 steps. Type F is abrupt, with no precursor.

## 4. Type N vs. type F
| | Type N | Type F |
|---|---|---|
| Loss floor | ~1e-8 | 2e-4 to 9e-4 (set by wd) |
| √v̂ vs. eps | √v̂ ≪ eps | √v̂ ≥ eps |
| Spikes | 1–2 | a cycle every 4–11k steps |
| Peak loss | 8–19 | 0.6–3 |

## 5. Per-step criterion
- The criterion frac_low × grad_share fails in both directions:
  - It is 1.0 for both the type-N spikers and the non-spikers.
  - It is about 0 for the type-F spikers at eps ≤ 1e-7.

## Recommendation
Re-run the grids with per-step minibatch loss and ‖g‖ logged, about 0.3 MB per run, and count spikes from those instead of from snapshots.
