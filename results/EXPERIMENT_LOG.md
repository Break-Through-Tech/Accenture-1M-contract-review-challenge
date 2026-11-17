# Experiment log: transformer tuning (October)

Setup for all rows: `bert-base-uncased`, 41 categories, 512-token chunks with 64-token overlap, 7,944 train / 2,317 validation chunks (84 validation contracts), seed 42. Test split not used. Metrics are on validation.

Caveat: "val-tuned" numbers pick thresholds on the same validation chunks they score, so they run about 0.03-0.05 F1 above "held-out" numbers (thresholds fit on other contracts, 5x5 CV).

## Summary

| # | Knob | Result | Verdict |
|---|---|---|---|
| 1 | Learning rate, epochs, empty-chunk sampling | Macro AP 0.415 -> 0.514 | Big gain; changes were bundled, so not attributable to one knob |
| 2 | Class weights | Weighted loss used in all runs; unweighted not run | Open |
| 3 | Threshold strategy | Per-category max-F1 best on val, but drops ~0.03-0.06 macro F1 held-out | Use held-out numbers for reporting |
| 4 | Blend baseline + transformer | Held-out chunk macro F1 0.503 / 0.481 -> 0.531 (50/50) | Beats both models; best result so far |

Chunk size was not varied (fixed at 512/64 throughout).

## Exp 1: learning rate, epochs, empty-chunk sampling

Source: `results/transformer_v1_run1/`, `results/transformer_v1/`.

| | Run 1 | Run 2 (v1) |
|---|---|---|
| Learning rate | 2e-5 all layers | 3e-5 encoder, 1e-3 classifier head |
| Epochs | 4 | 6 |
| Empty chunks per epoch | all | fresh random 25% |
| Chunk macro AP | 0.415 | 0.514 |
| Chunk macro F1 (val-tuned) | 0.466 | 0.550 |
| Chunk micro F1 (val-tuned) | 0.547 | 0.660 |
| Contract macro F1 (val-tuned) | 0.612 | 0.706 |
| Train time | 222 min | 2,467 min (laptop slept mid-run) |

Run 2 validation AP by epoch: 0.284, 0.418, 0.430, 0.484, 0.504, 0.514. Still rising at epoch 6, so more epochs are likely to help.
Limitation: three things changed together, so the gain cannot be split between them.

## Exp 2: class weights

Loss is `BCEWithLogitsLoss` with `pos_weight = sqrt(neg/pos)` per category in every full run. `scripts/transformer_multilabel.py` offers `--class-weight none`, but only a 5-step weighted smoke run exists (`results/transformer_multilabel/weighted_smoke/`), which shows the path works and nothing about quality. No weighted-vs-unweighted comparison has been run. It needs a full GPU run (run 2 took about 41 hours on an M4/MPS).

## Exp 3: threshold strategy

Source: `results/threshold_experiments/`. Chunk-level macro F1 on transformer v1 validation predictions:

| Strategy | Same validation set | Held-out (5-fold by contract) |
|---|---|---|
| fixed 0.5 | 0.470 | n/a |
| global macro-F1 threshold | 0.497 | 0.492 |
| per-category max F1 | 0.550 | 0.485 |
| per-category max F1 + rare fallback (<10 positives) | 0.533 | 0.481 |
| per-category recall >= 0.80 | 0.402 (recall 0.82) | 0.381 (recall 0.76) |

Per-category thresholds look best on validation but the gain does not hold up held-out; a single global threshold is as good. The recall-80 rule gets high recall at the cost of precision (chunk micro precision 0.19).

## Exp 4: blend baseline and transformer probabilities

New, no retraining. `python scripts/ensemble_experiment.py` -> `results/ensemble_experiment/`. Held-out thresholds (per-category max F1 with rare fallback), `w` = weight on the transformer:

| w | Chunk micro F1 | Chunk macro F1 | Contract micro F1 | Contract macro F1 | Chunk macro AP |
|---|---|---|---|---|---|
| 0 (baseline) | 0.628 | 0.503 | 0.808 | 0.644 | 0.551 |
| 0.3 | 0.646 | 0.529 | 0.819 | 0.668 | 0.576 |
| 0.5 | 0.667 | 0.531 | 0.833 | 0.666 | 0.572 |
| 0.7 | 0.652 | 0.506 | 0.820 | 0.650 | 0.560 |
| 1 (transformer v1) | 0.623 | 0.481 | 0.801 | 0.638 | 0.514 |

The models win on different categories, so averaging beats both: +0.03-0.05 chunk macro F1 over the transformer alone and +0.03 over the baseline at w=0.3-0.5. The 0 and 1 rows match `results/evaluation/SUMMARY.md`. The weight was chosen on the same validation set, so a small optimistic bias remains.

## Next

Unweighted-loss run (Exp 2), longer training or `legal-bert-base-uncased`, a chunk-size comparison, then final numbers on the test split once.
