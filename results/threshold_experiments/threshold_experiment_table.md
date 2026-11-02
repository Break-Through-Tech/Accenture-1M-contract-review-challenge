### Chunk-level

| strategy | scoring | level | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | same validation set | chunk | 0.469 | 0.751 | 0.577 | 0.393 | 0.618 | 0.470 |
| global_macro_f1 | same validation set | chunk | 0.575 | 0.692 | 0.628 | 0.481 | 0.542 | 0.497 |
| global_macro_f1 | 5-fold by contract (held-out) | chunk | 0.584 | 0.681 | 0.629 | 0.483 | 0.527 | 0.492 |
| per_category_max_f1 | same validation set | chunk | 0.642 | 0.679 | 0.660 | 0.569 | 0.578 | 0.550 |
| per_category_max_f1 | 5-fold by contract (held-out) | chunk | 0.598 | 0.631 | 0.614 | 0.507 | 0.504 | 0.485 |
| per_category_max_f1_rare_fallback | same validation set | chunk | 0.656 | 0.674 | 0.665 | 0.559 | 0.539 | 0.533 |
| per_category_max_f1_rare_fallback | 5-fold by contract (held-out) | chunk | 0.616 | 0.629 | 0.622 | 0.501 | 0.483 | 0.481 |
| per_category_recall_80 | same validation set | chunk | 0.187 | 0.808 | 0.304 | 0.338 | 0.817 | 0.402 |
| per_category_recall_80 | 5-fold by contract (held-out) | chunk | 0.179 | 0.785 | 0.292 | 0.322 | 0.761 | 0.381 |

### Contract-level

| strategy | scoring | level | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | same validation set | contract | 0.693 | 0.925 | 0.792 | 0.559 | 0.826 | 0.650 |
| global_macro_f1 | same validation set | contract | 0.762 | 0.885 | 0.819 | 0.611 | 0.736 | 0.660 |
| global_macro_f1 | 5-fold by contract (held-out) | contract | 0.768 | 0.876 | 0.818 | 0.610 | 0.717 | 0.652 |
| per_category_max_f1 | same validation set | contract | 0.786 | 0.864 | 0.823 | 0.677 | 0.789 | 0.706 |
| per_category_max_f1 | 5-fold by contract (held-out) | contract | 0.764 | 0.826 | 0.794 | 0.639 | 0.705 | 0.650 |
| per_category_max_f1_rare_fallback | same validation set | contract | 0.807 | 0.855 | 0.831 | 0.666 | 0.726 | 0.684 |
| per_category_max_f1_rare_fallback | 5-fold by contract (held-out) | contract | 0.787 | 0.822 | 0.804 | 0.628 | 0.667 | 0.640 |
| per_category_recall_80 | same validation set | contract | 0.485 | 0.948 | 0.642 | 0.475 | 0.960 | 0.565 |
| per_category_recall_80 | 5-fold by contract (held-out) | contract | 0.483 | 0.932 | 0.636 | 0.466 | 0.929 | 0.552 |
