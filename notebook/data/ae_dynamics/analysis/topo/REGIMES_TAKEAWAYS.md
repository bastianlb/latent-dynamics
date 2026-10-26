# Regimes: affine share R1 and scale-free topology, classifier vs AE (N = 1000, test probe, mean of 3 seeds)

Epoch is the run's own `epoch` field (step/8 at N = 1000, so step 100 = 12.5 epochs, not 12.8). Topology uses mean-pairwise-normalised latents; H1 quantities are on the 66 masked snapshots (step 0 included). "first" = step-0 snapshot (R1: first interval end, ~0.12 epoch); values read at the snapshot nearest in log-epoch.

## Regime markers (epochs, mean over 3 seeds)
| | MNIST | FM |
|---|---|---|
| MLP AE gap onset (recon_test/recon_train - 1 > 0.10) | 30.7 | 36.2 |
| MLP AE test-recon minimum | 119.5 | 150.8 |
| classifier train acc >= 0.999 | 20.1 | 40.6 |
| classifier test-acc maximum | 2337 (flat, noisy) | 34.2 |
| linear AE gap onset | never reaches 10% | never reaches 10% |
| linear AE test-recon minimum | 1186 | 2288 |
Fixed: 1 epoch, step 100 = 12.5 epochs. Seeds are in regime_numbers.json.

## Per metric (first / 1 ep / gap onset / test min / end), MNIST then FM
**Affine share R1**
- Classifier: MNIST 0.73 / 0.97 / 0.97 / 0.98 / 0.77; FM 0.85 / 0.99 / 0.92 / 0.98 / 1.00. Rises within 1 epoch and stays near 0.97-0.99 through all later regimes (final MNIST point 0.77 is a last-snapshot drop, see caveats).
- MLP AE: MNIST 0.91 / 0.99 / 0.92 / 0.87 / 0.83; FM 0.91 / 1.00 / 0.89 / 0.87 / 0.81. Peak ~1.0 near 1 epoch, then declines from about step 100 to 0.7-0.8 (minimum 0.71 MNIST, 0.75 FM near epoch 630).
- Linear AE: MNIST 0.97 / 0.87 / 0.89 / 0.49 / 0.86; FM 0.96 / 0.88 / 0.92 / 0.79 / 0.91. Dips to 0.21 (MNIST, epoch ~210) and 0.69 (FM, epoch ~310), then partly recovers.
- Classifier stays high; both AEs decline after ~step 100. MNIST and FM agree qualitatively; the linear-AE dip is much deeper on MNIST.

**H1 persistence entropy**
- Classifier: MNIST 6.81 / 6.20 / 6.18 / 6.18 / 5.92; FM 6.50 / 5.55 / 5.94 / 5.93 / 5.92. Falls within ~1 epoch, then flat (FM recovers slightly from 5.51 at ep 1.2).
- MLP AE: MNIST 6.81 / 6.11 / 6.45 / 6.49 / 6.47; FM 6.50 / 5.38 / 5.94 / 6.04 / 6.07. Same initial fall as classifier to a minimum at epoch ~4 (5.98 MNIST, 5.28 FM), then rises back to 6.5 / 6.1.
- Linear AE: MNIST 6.80 / 6.79 / 6.36 / 6.50 / 6.51 (flat until ~epoch 3, then falls to 6.36 at ep 32, rises); FM 6.36 / 6.49 / 5.78 / 5.90 / 5.90.
- Models coincide before 1 epoch (classifier and MLP AE) and differ after: classifier plateau stays low, MLP AE rebounds to near its start value on MNIST. Datasets agree on shape; FM is ~0.5 lower throughout.

**H1 lifetime sum**
- Classifier: MNIST 38.3 / 19.0 / 19.3 / 19.5 / 13.4; FM 24.6 / 6.6 / 10.5 / 11.4 / 13.2. Drops by ~2x to 4x in 1 epoch, then flat or slowly rising (end value affected by last snapshot).
- MLP AE: MNIST 38.3 / 10.6 / 31.4 / 33.5 / 29.8; FM 24.6 / 5.1 / 14.5 / 15.2 / 15.1. Drops to a minimum at ~1 epoch (below classifier), then climbs back to near the initial value (MNIST) or 0.6x of it (FM).
- Linear AE: MNIST 39.6 / 37.5 / 27.5 / 33.4 / 34.2; FM 19.7 / 23.8 / 10.0 / 13.2 / 14.4. Unchanged through ~3 epochs, then falls (minimum 26.3 MNIST at ep 21, 8.7 FM at ep 16), partial recovery.
- At late times FM classifier, MLP AE and linear AE lifetime sums are similar (13-15); on MNIST they separate (classifier ~19, AEs 30-34).

**PMag (fixed window)**
- Classifier: MNIST 264 / 107 / 127 / 128 / 109; FM 217 / 38 / 67 / 76 / 98. Min ~99 (MNIST, ep 1.8) / 38 (FM, ep 1.2), then a modest rise.
- MLP AE: MNIST 264 / 57 / 194 / 212 / 228; FM 217 / 29 / 112 / 130 / 159. Min at epoch 1 (57 / 29), then rises to or above its starting level (MNIST 228 vs 264; FM 159 vs 217) with no plateau by epoch 2500 on FM.
- Linear AE: MNIST 260 / 284 / 174 / 221 / 233; FM 194 / 236 / 85 / 117 / 132. Flat or slightly up to ~epoch 1, minimum at epoch ~16-21 (165 MNIST, 68 FM), then rises.

## Where the models coincide / differ
- Coincide: the classifier and the MLP AE move together in the first ~1 epoch on all three topological summaries (entropy 6.8 to ~6.1-6.2, lifetime sum and PMag falling), and both have R1 near 0.97-1.0 at 1 epoch.
- Differ: after ~1 epoch the MLP AE metrics reverse (entropy 6.1 to 6.5, lifetime sum 10.6 to 30+, PMag 57 to ~210 on MNIST); the classifier stays near its 1-epoch value (flat within ~0.1 entropy, lifetime sum and PMag changing by <25% on MNIST). The reversal in the MLP AE spans epochs ~1 to ~100, i.e. it begins before gap onset (30-36 epochs).
- R1 separates the models from step ~100: classifier ~0.98, MLP AE falls to ~0.8, linear AE falls to 0.2-0.7.
- Linear AE differs from the MLP AE in timing: its topology does not change in the first ~1-3 epochs, and its minima come at epochs ~16-50, later than the MLP AE (epoch ~1-4). It never reaches a 10% test/train recon gap.
- MNIST vs FM: same direction and ordering of changes for every model and metric; FM has lower entropy/lifetime sum/PMag overall, FM classifier metrics rise slightly after 1 epoch (e.g. lifetime sum 6.6 to 13), MNIST classifier metrics do not.

## Caveats
- The classifier cache's last snapshot (step 20000) is lower than the preceding one in entropy, lifetime sum, PMag and MNIST R1 (e.g. MNIST entropy 6.17 to 5.92, R1 0.98 to 0.77); this appears as a sharp last-point drop in the figure. Not investigated.
- Train-probe topology was not computed (test pass took ~35 min). Regime markers for N = full: MLP AE has no 10% gap onset.
