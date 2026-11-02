"""Threshold experiments on validation predictions (issue #20).

Reads the per-chunk validation probabilities saved by a trained model and compares
ways of turning them into yes/no clause predictions: a fixed 0.5, one global
threshold, and per-category thresholds tuned for F1 or for a recall target.

Thresholds tuned and scored on the same validation set are optimistic, so each tuned
strategy is also scored with contract-grouped cross-validation: the thresholds are
fitted on the other contracts and applied to the held-out ones. The test split is
never read.

    python scripts/threshold_experiments.py
    python scripts/threshold_experiments.py --predictions results/other_run/val_predictions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = PROJECT_DIR / "results" / "transformer_v1" / "val_predictions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "threshold_experiments"

DEFAULT_THRESHOLD = 0.5
RECALL_TARGET = 0.80
MIN_POSITIVES = 10
GLOBAL_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
N_FOLDS = 5
N_REPEATS = 5
SEED = 42

PROB_PREFIX = "prob__"
LABEL_PREFIX = "label__"


# --- data -------------------------------------------------------------------------

def load_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return categories, labels Y, probabilities P (chunks x categories), contract ids."""
    frame = pd.read_csv(path)
    categories = [c[len(PROB_PREFIX):] for c in frame.columns if c.startswith(PROB_PREFIX)]
    Y = frame[[LABEL_PREFIX + c for c in categories]].to_numpy(dtype=int)
    P = frame[[PROB_PREFIX + c for c in categories]].to_numpy(dtype=float)
    return categories, Y, P, frame["contract_id"].to_numpy()


def contract_codes(contract_ids: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(contract_ids)
    return codes, len(uniques)


def any_per_contract(values: np.ndarray, codes: np.ndarray, n_contracts: int) -> np.ndarray:
    """Contract-level OR (or max) of a chunk x category array."""
    out = np.zeros((n_contracts, values.shape[1]), dtype=values.dtype)
    np.maximum.at(out, codes, values)
    return out


# --- metrics ----------------------------------------------------------------------

def confusion_counts(Y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = pred.astype(bool)
    positive = Y.astype(bool)
    tp = (pred & positive).sum(axis=0)
    fp = (pred & ~positive).sum(axis=0)
    fn = (~pred & positive).sum(axis=0)
    return tp, fp, fn


def _prf(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def summarise(Y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Micro and macro P/R/F1, macro skipping categories with no positives."""
    tp, fp, fn = confusion_counts(Y, pred)
    micro_p, micro_r, micro_f1 = (float(x) for x in _prf(tp.sum(), fp.sum(), fn.sum()))
    precision, recall, f1 = _prf(tp, fp, fn)
    has_positive = Y.sum(axis=0) > 0
    return {
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "macro_precision": float(precision[has_positive].mean()),
        "macro_recall": float(recall[has_positive].mean()),
        "macro_f1": float(f1[has_positive].mean()),
    }


def category_metrics(categories, Y, pred, thresholds) -> pd.DataFrame:
    tp, fp, fn = confusion_counts(Y, pred)
    precision, recall, f1 = _prf(tp, fp, fn)
    return pd.DataFrame({
        "category": categories, "threshold": thresholds, "positives": Y.sum(axis=0),
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
    })


# --- threshold rules --------------------------------------------------------------

def fit_max_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises F1; 0.5 when the category has no positives."""
    if y.sum() == 0:
        return DEFAULT_THRESHOLD
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def fit_recall_target(y: np.ndarray, p: np.ndarray, target: float = RECALL_TARGET) -> float:
    """Highest threshold that still reaches the recall target (so precision is as high as possible)."""
    if y.sum() == 0:
        return DEFAULT_THRESHOLD
    _, recall, thresholds = precision_recall_curve(y, p)
    reachable = np.flatnonzero(recall[:-1] >= target)
    return float(thresholds[reachable[-1]]) if len(reachable) else float(thresholds[0])


def fit_global(Y: np.ndarray, P: np.ndarray) -> float:
    """One threshold for every category, chosen to maximise macro F1."""
    scores = [summarise(Y, P >= t)["macro_f1"] for t in GLOBAL_GRID]
    return float(GLOBAL_GRID[int(np.argmax(scores))])


FitFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _per_category(rule: Callable[[np.ndarray, np.ndarray], float]) -> FitFn:
    return lambda Y, P: np.array([rule(Y[:, j], P[:, j]) for j in range(Y.shape[1])])


def _fixed(Y, P):
    return np.full(Y.shape[1], DEFAULT_THRESHOLD)


def _global(Y, P):
    return np.full(Y.shape[1], fit_global(Y, P))


def _max_f1_with_fallback(Y, P):
    """Per-category F1 threshold; categories with few positives use the global threshold."""
    thresholds = _per_category(fit_max_f1)(Y, P)
    rare = Y.sum(axis=0) < MIN_POSITIVES
    thresholds[rare] = fit_global(Y, P)
    return thresholds


STRATEGIES: dict[str, FitFn] = {
    "fixed_0.5": _fixed,
    "global_macro_f1": _global,
    "per_category_max_f1": _per_category(fit_max_f1),
    "per_category_max_f1_rare_fallback": _max_f1_with_fallback,
    f"per_category_recall_{int(RECALL_TARGET * 100)}": _per_category(fit_recall_target),
}


# --- evaluation -------------------------------------------------------------------

def level_metrics(Y, pred, codes, n_contracts) -> dict[str, dict[str, float]]:
    return {
        "chunk": summarise(Y, pred),
        "contract": summarise(any_per_contract(Y, codes, n_contracts),
                              any_per_contract(pred.astype(int), codes, n_contracts)),
    }


def cross_validated_predictions(fit: FitFn, Y, P, codes, n_contracts, seed: int) -> np.ndarray:
    """Predictions where every contract is scored with thresholds fitted on the other folds."""
    folds = np.random.default_rng(seed).permutation(n_contracts) % N_FOLDS
    chunk_fold = folds[codes]
    pred = np.zeros(Y.shape, dtype=bool)
    for fold in range(N_FOLDS):
        held_out = chunk_fold == fold
        thresholds = fit(Y[~held_out], P[~held_out])
        pred[held_out] = P[held_out] >= thresholds
    return pred


def run_experiments(categories, Y, P, contract_ids):
    codes, n_contracts = contract_codes(contract_ids)
    rows, thresholds_by_strategy, per_category = [], {}, []

    for name, fit in STRATEGIES.items():
        thresholds = fit(Y, P)
        thresholds_by_strategy[name] = thresholds
        pred = P >= thresholds
        metrics = level_metrics(Y, pred, codes, n_contracts)
        for level, values in metrics.items():
            rows.append({"strategy": name, "scoring": "same validation set", "level": level,
                         **values, "micro_f1_std": np.nan, "macro_f1_std": np.nan})
        cat = category_metrics(categories, Y, pred, thresholds)
        cat.insert(0, "strategy", name)
        per_category.append(cat)

        if name == "fixed_0.5":  # nothing is tuned, so nothing to cross-validate
            continue
        repeats = [level_metrics(Y, cross_validated_predictions(fit, Y, P, codes, n_contracts,
                                                                  SEED + r), codes, n_contracts)
                   for r in range(N_REPEATS)]
        for level in ("chunk", "contract"):
            frame = pd.DataFrame([r[level] for r in repeats])
            rows.append({"strategy": name, "scoring": f"{N_FOLDS}-fold by contract (held-out)",
                         "level": level, **frame.mean().to_dict(),
                         "micro_f1_std": frame["micro_f1"].std(ddof=1),
                         "macro_f1_std": frame["macro_f1"].std(ddof=1)})

    table = pd.DataFrame(rows)
    thresholds = pd.DataFrame(thresholds_by_strategy, index=categories)
    thresholds.insert(0, "positive_chunks", Y.sum(axis=0))
    thresholds.insert(1, "positive_contracts", any_per_contract(Y, codes, n_contracts).sum(axis=0))
    thresholds.index.name = "category"
    return table, thresholds.reset_index(), pd.concat(per_category, ignore_index=True)


# --- output -----------------------------------------------------------------------

def markdown_table(table: pd.DataFrame) -> str:
    columns = ["strategy", "scoring", "level", "micro_precision", "micro_recall", "micro_f1",
               "macro_precision", "macro_recall", "macro_f1"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in table[columns].iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(out: Path, table, thresholds, per_category, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    order = {name: i for i, name in enumerate(STRATEGIES)}
    table = table.assign(_order=table["strategy"].map(order),
                         _held_out=table["scoring"].str.contains("held-out")).sort_values(
        ["level", "_order", "_held_out"]).drop(columns=["_order", "_held_out"])
    table.round(4).to_csv(out / "threshold_experiment_table.csv", index=False)
    (out / "threshold_experiment_table.md").write_text(
        "\n\n".join(f"### {level.capitalize()}-level\n\n"
                    + markdown_table(table[table["level"] == level])
                    for level in ("chunk", "contract")) + "\n")
    thresholds.round(4).to_csv(out / "per_category_thresholds.csv", index=False)
    per_category.round(4).to_csv(out / "per_category_metrics_chunk_level.csv", index=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    categories, Y, P, contract_ids = load_predictions(args.predictions)
    table, thresholds, per_category = run_experiments(categories, Y, P, contract_ids)
    config = {
        "issue": 20,
        "predictions": str(args.predictions.relative_to(PROJECT_DIR)
                           if args.predictions.is_absolute() else args.predictions),
        "chunks": int(len(Y)), "contracts": int(len(set(contract_ids))),
        "categories": len(categories),
        "strategies": list(STRATEGIES),
        "recall_target": RECALL_TARGET, "min_positives_for_own_threshold": MIN_POSITIVES,
        "global_grid": [float(GLOBAL_GRID[0]), float(GLOBAL_GRID[-1]), 0.01],
        "cross_validation": {"folds": N_FOLDS, "repeats": N_REPEATS, "seed": SEED,
                             "unit": "contract"},
        "contract_rule": "contract flagged if any chunk is predicted positive",
        "test_set": "not used",
    }
    write_outputs(args.output_dir, table, thresholds, per_category, config)
    print((args.output_dir / "threshold_experiment_table.md").read_text())
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
