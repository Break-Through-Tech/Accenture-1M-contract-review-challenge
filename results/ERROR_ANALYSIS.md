# Error analysis: transformer v1 vs TF-IDF baseline

Reproduce: `python scripts/error_analysis.py` (no retraining; writes `results/error_analysis/`).

Scope: the 2,317 validation chunks (84 contracts), saved predictions of both models, per-category max-F1 thresholds with the rare-category fallback fitted on validation. Counts are therefore a little optimistic for both models. Test split not used. "Weak categories" are the four top-priority categories where the transformer trails the baseline (`results/evaluation/SUMMARY.md`).

## 1. Where the transformer loses (weak categories)

Chunk-level counts (`weak_categories_confusion.csv`):

| Category | Positives | Baseline TP / FP / FN | Transformer TP / FP / FN |
|---|---|---|---|
| IP Ownership Assignment | 46 | 33 / 14 / 13 | 25 / 12 / 21 |
| Termination for Convenience | 46 | 30 / 31 / 16 | 18 / 20 / 28 |
| Exclusivity | 53 | 32 / 36 / 21 | 27 / 50 / 26 |
| Minimum Commitment | 55 | 28 / 22 / 27 | 31 / 42 / 24 |

Over these four categories the transformer has 99 misses vs 77 and 124 false alarms vs 103. The loss differs by category:

- **IP Ownership Assignment, Termination for Convenience:** mostly lost recall (8 and 12 more misses).
- **Exclusivity, Minimum Commitment:** recall is similar or slightly better, but there are many more false alarms (+14 and +20).
- The errors are spread over many contracts (the most error-heavy contracts contribute 8-12 errors each), so no single bad contract explains the gap.

## 2. Clauses split across chunk edges are missed about twice as often

A positive chunk is one that overlaps an annotated span. For 336 of 1,849 positive (chunk, category) pairs (18%) the span runs past the chunk edge, so the chunk holds only a fragment (`miss_rate_by_span_coverage.csv`):

| Model | Span fully in chunk: miss rate | Span cut by chunk edge: miss rate |
|---|---|---|
| Baseline | 0.29 (442 of 1,513) | 0.61 (204 of 336) |
| Transformer | 0.28 (425 of 1,513) | 0.53 (178 of 336) |

This is the largest single error source we found, and it affects both models. Part of it is a labelling artefact: a chunk containing a few words of a clause counts as positive, and no model can be expected to flag it. Contract-level scores (a contract counts if any chunk fires) are less affected, which is consistent with contract micro F1 (0.80) being well above chunk micro F1 (0.62).

## 3. Span length

Miss rate for chunks that contain the whole span (`miss_rate_by_span_length.csv`):

| Span length | Baseline | Transformer |
|---|---|---|
| under 100 chars | 0.32 | 0.26 |
| 100-300 | 0.34 | 0.32 |
| 300-1000 | 0.21 | 0.26 |
| over 1000 (n=20) | 0.30 | 0.10 |

The transformer is better on short and very long spans and worse on mid-length ones (300-1000 chars, +0.05). Differences are small and the long-span row has only 20 chunks, so treat this as a weak signal.

## 4. What the individual errors look like (`examples.csv`)

- **Misses at high scores.** Several transformer misses score 0.56-0.81 (for example a plain "each party may terminate for convenience after Contract Year 2" at 0.81, and "University will assign the IND to ArTara" at 0.79). The per-category thresholds for these categories sit above those scores. Many false alarms score 0.8-0.97, so the max-F1 threshold is pushed high and true positives under it are lost. The transformer's scores look overconfident; calibration would likely help. This is our reading of the pattern, not tested.
- **False alarms are topical neighbours.** Exclusivity fires on definitions and product lists, Termination for Convenience on payment and termination-for-cause text, Minimum Commitment on lead times and support obligations, and IP Ownership Assignment on ownership or "control" language that is not an assignment. Some of these (such as "shall be owned by Ono") look like real ownership clauses that CUAD did not annotate in that category, so part of the false-alarm count is label inconsistency rather than model error.
- **Misses are often short, sentence-level clauses** embedded in long operative sections, or clauses with redacted terms ("[ * ]"), which remove the cue words.

## 5. Suggested next steps

1. Reduce edge effects: use more overlap or merge neighbouring chunk predictions, and label a chunk positive only when most of the span is inside it.
2. Calibrate probabilities (for example per-category temperature or Platt scaling on validation) before choosing thresholds.
3. Retrain with the unweighted comparison from the experiment log and with `legal-bert-base-uncased`, then re-run this analysis on the four weak categories.
4. Use the blend of both models (see `EXPERIMENT_LOG.md`, experiment 4); the two models err on different chunks.
5. Review the false alarms in `examples.csv` with the advisor to separate label noise from model error.
