"""Evaluation: transformer v1 vs the TF-IDF baseline on validation (issue: Evaluation).

Both models are scored on the same validation chunks with the same threshold rule,
per-category max-F1 thresholds with a global fallback for rare categories
(``per_category_max_f1_rare_fallback`` from ``threshold_experiments.py``). The
headline numbers are cross-validated by contract: each contract is scored with
thresholds fitted on the other contracts, so the thresholds never see the chunks
they are scored on. Numbers with thresholds tuned on the full validation set are
reported alongside for comparison with earlier results. The test split is never read.

The baseline's validation predictions are not saved by ``baseline_tfidf.py``, so the
first run retrains it (about 2 minutes) and saves them in the same format as the
transformer's ``val_predictions.csv``. Later runs reuse that file.

    python scripts/evaluate.py
    python scripts/evaluate.py --refresh-baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import threshold_experiments as te

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRANSFORMER_PREDICTIONS = PROJECT_DIR / "results" / "transformer_v1" / "val_predictions.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "evaluation"
BASELINE_PREDICTIONS = OUTPUT_DIR / "baseline_val_predictions.csv"

STRATEGY = "per_category_max_f1_rare_fallback"
MODELS = ("baseline", "transformer")
KEY = ["contract_id", "chunk_index"]

# The EDA shortlist (data/eda/category_profile.csv) in the team's preliminary priority
# order, most important first. Subject to change.
PRIORITY = [
    "Cap on Liability",
    "IP Ownership Assignment",
    "License Grant",
    "Termination for Convenience",
    "Exclusivity",
    "Non-Compete",
    "Minimum Commitment",
    "Change of Control",
    "Anti-Assignment",
    "Insurance",
    "Revenue/Profit Sharing",
    "Post-Termination Services",
    "Audit Rights",
    "Non-Transferable License",
]

COLORS = {"baseline": "#2a78d6", "transformer": "#eb6834"}
INK, INK_SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


# --- predictions ------------------------------------------------------------------

def baseline_val_predictions(categories: list[str]) -> pd.DataFrame:
    """Retrain the baseline exactly as baseline_tfidf.py does and score validation chunks."""
    import baseline_tfidf as b
    from load_dataset import load_annotations

    result = b.run_baseline(categories)
    val = result.chunks.loc[result.chunks["split"] == "val"].reset_index(drop=True)
    P = b.predict_proba(result.models, result.vectorizer.transform(val["text"]), categories)
    Y = b.label_chunks(val, load_annotations(), categories)
    return pd.concat([val.loc[:, ["contract_id", "chunk_index", "start", "end"]],
                      P.add_prefix(te.PROB_PREFIX), Y.add_prefix(te.LABEL_PREFIX)], axis=1)


def load_aligned(transformer_path: Path, baseline_path: Path, refresh: bool):
    """Both models' probabilities on the same chunks, in the same row order."""
    categories, _, _, _ = te.load_predictions(transformer_path)
    if refresh or not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_val_predictions(categories).round(6).to_csv(baseline_path, index=False)

    frames = {"transformer": pd.read_csv(transformer_path), "baseline": pd.read_csv(baseline_path)}
    frames = {m: f.sort_values(KEY).reset_index(drop=True) for m, f in frames.items()}
    t, b = frames["transformer"], frames["baseline"]
    if not t[KEY].equals(b[KEY]):
        raise ValueError("baseline and transformer predictions cover different chunks")
    labels = [te.LABEL_PREFIX + c for c in categories]
    if not np.array_equal(t[labels].to_numpy(), b[labels].to_numpy()):
        raise ValueError("baseline and transformer chunks have different labels")

    Y = t[labels].to_numpy(dtype=int)
    P = {m: frames[m][[te.PROB_PREFIX + c for c in categories]].to_numpy(dtype=float)
         for m in MODELS}
    return categories, Y, P, t["contract_id"].to_numpy()


# --- metrics ----------------------------------------------------------------------

def predictions_by_scoring(Y, P, codes, n_contracts) -> dict[str, list[np.ndarray]]:
    """Thresholded predictions: tuned on the full set, and one per cross-validation repeat."""
    fit = te.STRATEGIES[STRATEGY]
    return {
        "same validation set": [P >= fit(Y, P)],
        "held-out": [te.cross_validated_predictions(fit, Y, P, codes, n_contracts, te.SEED + r)
                     for r in range(te.N_REPEATS)],
    }


def per_category(categories, Y, P, preds: list[np.ndarray]) -> pd.DataFrame:
    """Per-category P/R/F1 averaged over repeats, plus threshold-free average precision."""
    metrics = ["precision", "recall", "f1"]
    runs = [te.category_metrics(categories, Y, p, np.nan).loc[:, metrics].to_numpy()
            for p in preds]
    out = pd.DataFrame(np.mean(runs, axis=0), columns=metrics)
    out.insert(0, "category", categories)
    out.insert(1, "positives", Y.sum(axis=0))
    out["avg_precision"] = [average_precision_score(Y[:, j], P[:, j]) if Y[:, j].any() else np.nan
                            for j in range(Y.shape[1])]
    return out


def evaluate(categories: list[str], Y: np.ndarray, P: dict[str, np.ndarray],
             contract_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    codes, n_contracts = te.contract_codes(contract_ids)
    Yc = te.any_per_contract(Y, codes, n_contracts)
    overall, chunk_rows, contract_rows = [], [], []

    for model in MODELS:
        for scoring, preds in predictions_by_scoring(Y, P[model], codes, n_contracts).items():
            contract_preds = [te.any_per_contract(p.astype(int), codes, n_contracts) for p in preds]
            for level, y, ps in (("chunk", Y, preds), ("contract", Yc, contract_preds)):
                runs = pd.DataFrame([te.summarise(y, p) for p in ps])
                overall.append({"model": model, "scoring": scoring, "level": level,
                                **runs.mean().to_dict()})
            chunk_rows.append(per_category(categories, Y, P[model], preds)
                              .assign(model=model, scoring=scoring))
            contract_rows.append(per_category(categories, Yc,
                                              te.any_per_contract(P[model], codes, n_contracts),
                                              contract_preds)
                                 .assign(model=model, scoring=scoring))

    return (pd.DataFrame(overall), pd.concat(chunk_rows, ignore_index=True),
            pd.concat(contract_rows, ignore_index=True))


def side_by_side(per_cat: pd.DataFrame, scoring: str = "held-out") -> pd.DataFrame:
    """One row per category: baseline and transformer metrics and the change."""
    metrics = ["precision", "recall", "f1", "avg_precision"]
    rows = per_cat.loc[per_cat["scoring"] == scoring]
    wide = rows.pivot(index="category", columns="model", values=metrics)
    wide.columns = [f"{model}_{metric}" for metric, model in wide.columns]
    wide = wide.loc[:, [f"{m}_{metric}" for m in MODELS for metric in metrics]]
    for metric in ("f1", "recall", "avg_precision"):
        wide[f"change_{metric}"] = wide[f"transformer_{metric}"] - wide[f"baseline_{metric}"]
    positives = rows.drop_duplicates(subset="category").set_index("category")["positives"]
    wide.insert(0, "positives", positives)
    return wide.loc[rows["category"].unique()].reset_index()


def priority_table(chunk_wide: pd.DataFrame, contract_wide: pd.DataFrame) -> pd.DataFrame:
    order = pd.DataFrame({"rank": range(1, len(PRIORITY) + 1), "category": PRIORITY})
    cols = ["category", "positives", "baseline_f1", "transformer_f1", "change_f1",
            "baseline_recall", "transformer_recall", "change_recall"]
    chunk = chunk_wide.loc[:, cols]
    contract = contract_wide.loc[:, cols].add_prefix("contract_").rename(
        columns={"contract_category": "category"})
    return order.merge(chunk, on="category", how="left").merge(contract, on="category", how="left")


# --- output -----------------------------------------------------------------------

def markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for _, row in frame.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_overall(overall: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    held = overall[overall["scoring"] == "held-out"]
    groups = [(level, avg) for level in ("chunk", "contract") for avg in ("micro", "macro")]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    height = 0.36
    for i, model in enumerate(MODELS):
        values = [held.loc[(held["model"] == model) & (held["level"] == lvl), f"{avg}_f1"].iloc[0]
                  for lvl, avg in groups]
        y = np.arange(len(groups)) + (i - 0.5) * (height + 0.04)
        ax.barh(y, values, height=height, color=COLORS[model], label=model.capitalize())
        for yy, v in zip(y, values):
            ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(np.arange(len(groups)), [f"{lvl.capitalize()} level, {avg} F1"
                                          for lvl, avg in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Transformer v1 vs TF-IDF baseline, validation F1", loc="left",
                 color=INK, fontsize=12)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_SECONDARY)
    fig.text(0.01, 0.01, "Thresholds: per-category max F1 with rare-category fallback, "
             "cross-validated by contract.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_per_category(wide: pd.DataFrame, path: Path, title: str, ordered: bool = False) -> None:
    """Dot plot of per-category F1, one row per category, a line joining the two models."""
    import matplotlib.pyplot as plt

    data = wide if ordered else wide.sort_values("transformer_f1", ascending=False)
    labels = data["category"].tolist()
    if "rank" in data:
        labels = [f"{r}. {c}" for r, c in zip(data["rank"], data["category"])]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(data) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.hlines(y, data[["baseline_f1", "transformer_f1"]].min(axis=1),
              data[["baseline_f1", "transformer_f1"]].max(axis=1), color="#c3c2b7", linewidth=2)
    for model in MODELS:
        ax.scatter(data[f"{model}_f1"], y, s=48, color=COLORS[model], label=model.capitalize(),
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (chunk level)", color=INK_SECONDARY)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              labelcolor=INK_SECONDARY, borderaxespad=0.2, handletextpad=0.3)
    ax.set_ylim(len(data) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def write_outputs(out: Path, overall: pd.DataFrame, chunk_cat: pd.DataFrame,
                  contract_cat: pd.DataFrame, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    overall = overall.sort_values(["level", "scoring", "model"], ascending=[True, True, True])
    overall.round(4).to_csv(out / "overall_metrics.csv", index=False)
    pd.concat([chunk_cat.assign(level="chunk"), contract_cat.assign(level="contract")]) \
        .round(4).to_csv(out / "per_category_metrics.csv", index=False)

    chunk_wide, contract_wide = side_by_side(chunk_cat), side_by_side(contract_cat)
    chunk_wide.round(4).to_csv(out / "comparison_chunk_level.csv", index=False)
    contract_wide.round(4).to_csv(out / "comparison_contract_level.csv", index=False)
    priority = priority_table(chunk_wide, contract_wide)
    priority.round(4).to_csv(out / "priority_categories.csv", index=False)

    cols = ["model", "scoring", "micro_precision", "micro_recall", "micro_f1",
            "macro_precision", "macro_recall", "macro_f1"]
    sections = [f"### {level.capitalize()}-level\n\n"
                + markdown(overall.loc[overall["level"] == level, cols].round(3))
                for level in ("chunk", "contract")]
    sections.append("### Priority categories (held-out, chunk level)\n\n" + markdown(
        priority.loc[:, ["rank", "category", "positives", "baseline_f1", "transformer_f1",
                         "change_f1", "baseline_recall", "transformer_recall"]].round(3)))
    (out / "metrics_table.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")

    plot_overall(overall, out / "overall_f1.png")
    plot_per_category(chunk_wide, out / "per_category_f1.png",
                      "Per-category F1, all 41 categories (held-out thresholds)")
    plot_per_category(priority, out / "priority_categories_f1.png",
                      "Per-category F1, priority categories in rank order", ordered=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transformer", type=Path, default=TRANSFORMER_PREDICTIONS)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PREDICTIONS)
    parser.add_argument("--refresh-baseline", action="store_true",
                        help="retrain the baseline even if its predictions are saved")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    categories, Y, P, contract_ids = load_aligned(args.transformer, args.baseline,
                                                  args.refresh_baseline)
    overall, chunk_cat, contract_cat = evaluate(categories, Y, P, contract_ids)
    config = {
        "predictions": {"transformer": str(args.transformer.relative_to(PROJECT_DIR)),
                        "baseline": str(args.baseline.relative_to(PROJECT_DIR))},
        "split": "val", "test_set": "not used",
        "chunks": int(len(Y)), "contracts": int(len(set(contract_ids))),
        "categories": len(categories),
        "threshold_strategy": STRATEGY,
        "min_positives_for_own_threshold": te.MIN_POSITIVES,
        "headline_scoring": {"name": "held-out", "folds": te.N_FOLDS,
                             "repeats": te.N_REPEATS, "seed": te.SEED, "unit": "contract"},
        "contract_rule": "contract flagged if any chunk is predicted positive",
        "avg_precision": "threshold-free, on the full validation set",
    }
    write_outputs(args.output_dir, overall, chunk_cat, contract_cat, config)
    print((args.output_dir / "metrics_table.md").read_text(encoding="utf-8"))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
