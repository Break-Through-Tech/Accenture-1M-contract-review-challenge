"""Experiment: average the baseline's and transformer v1's probabilities (no retraining).

Scores each blend on the same validation chunks with the same held-out threshold rule as
evaluate.py (per-category max F1, rare fallback, 5x5 CV by contract). Test split unused.

    python scripts/ensemble_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev
import threshold_experiments as te

OUT = ev.OUTPUT_DIR.parent / "ensemble_experiment"
WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]  # weight on the transformer; 0 = baseline, 1 = transformer


def main() -> None:
    cats, Y, P, ids = ev.load_aligned(ev.TRANSFORMER_PREDICTIONS, ev.BASELINE_PREDICTIONS, False)
    codes, n = te.contract_codes(ids)
    Yc = te.any_per_contract(Y, codes, n)
    rows = []
    for w in WEIGHTS:
        blend = w * P["transformer"] + (1 - w) * P["baseline"]
        preds = ev.predictions_by_scoring(Y, blend, codes, n)["held-out"]
        cpreds = [te.any_per_contract(p.astype(int), codes, n) for p in preds]
        row = {"transformer_weight": w}
        for level, y, ps in (("chunk", Y, preds), ("contract", Yc, cpreds)):
            m = pd.DataFrame([te.summarise(y, p) for p in ps]).mean()
            row.update({f"{level}_micro_f1": m["micro_f1"], f"{level}_macro_f1": m["macro_f1"],
                        f"{level}_macro_recall": m["macro_recall"]})
        ap = [ev.average_precision_score(Y[:, j], blend[:, j]) for j in range(Y.shape[1]) if Y[:, j].any()]
        row["chunk_macro_avg_precision"] = float(np.mean(ap))
        rows.append(row)
    df = pd.DataFrame(rows).round(4)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ensemble_results.csv", index=False)
    (OUT / "ensemble_results.md").write_text(ev.markdown(df) + "\n")
    print(ev.markdown(df))


if __name__ == "__main__":
    main()
