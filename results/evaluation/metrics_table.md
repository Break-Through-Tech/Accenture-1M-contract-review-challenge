### Chunk-level

| model | scoring | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|---|---|
| baseline | held-out | 0.643 | 0.614 | 0.628 | 0.563 | 0.489 | 0.503 |
| transformer | held-out | 0.620 | 0.626 | 0.623 | 0.503 | 0.481 | 0.481 |
| baseline | same validation set | 0.698 | 0.651 | 0.673 | 0.626 | 0.544 | 0.556 |
| transformer | same validation set | 0.656 | 0.674 | 0.665 | 0.559 | 0.539 | 0.533 |

### Contract-level

| model | scoring | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|---|---|
| baseline | held-out | 0.830 | 0.787 | 0.808 | 0.684 | 0.635 | 0.644 |
| transformer | held-out | 0.787 | 0.817 | 0.801 | 0.628 | 0.664 | 0.638 |
| baseline | same validation set | 0.851 | 0.812 | 0.831 | 0.722 | 0.685 | 0.683 |
| transformer | same validation set | 0.807 | 0.855 | 0.831 | 0.666 | 0.726 | 0.684 |

### Priority categories (held-out, chunk level)

| rank | category | positives | baseline_f1 | transformer_f1 | change_f1 | baseline_recall | transformer_recall |
|---|---|---|---|---|---|---|---|
| 1 | Cap on Liability | 101 | 0.722 | 0.749 | 0.028 | 0.622 | 0.646 |
| 2 | IP Ownership Assignment | 46 | 0.687 | 0.524 | -0.163 | 0.683 | 0.452 |
| 3 | License Grant | 124 | 0.724 | 0.786 | 0.062 | 0.765 | 0.750 |
| 4 | Termination for Convenience | 46 | 0.506 | 0.321 | -0.185 | 0.565 | 0.352 |
| 5 | Exclusivity | 53 | 0.498 | 0.369 | -0.129 | 0.570 | 0.423 |
| 6 | Non-Compete | 32 | 0.342 | 0.367 | 0.025 | 0.356 | 0.456 |
| 7 | Minimum Commitment | 55 | 0.483 | 0.432 | -0.051 | 0.549 | 0.429 |
| 8 | Change of Control | 42 | 0.418 | 0.448 | 0.030 | 0.310 | 0.400 |
| 9 | Anti-Assignment | 112 | 0.806 | 0.848 | 0.042 | 0.734 | 0.789 |
| 10 | Insurance | 48 | 0.787 | 0.759 | -0.029 | 0.704 | 0.779 |
| 11 | Revenue/Profit Sharing | 39 | 0.506 | 0.512 | 0.006 | 0.513 | 0.528 |
| 12 | Post-Termination Services | 84 | 0.352 | 0.514 | 0.162 | 0.381 | 0.657 |
| 13 | Audit Rights | 86 | 0.779 | 0.675 | -0.104 | 0.774 | 0.805 |
| 14 | Non-Transferable License | 48 | 0.614 | 0.566 | -0.048 | 0.638 | 0.696 |
