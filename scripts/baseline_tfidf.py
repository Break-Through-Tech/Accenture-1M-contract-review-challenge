"""TF-IDF + Logistic Regression baseline for chunk-level clause detection.

Contracts are cut into the same 512-token, 64-overlap ``bert-base-uncased``
chunks as ``notebooks/chunking.ipynb``, and each chunk inherits its contract's
split from ``splits.csv``.  A chunk is positive for a category if it overlaps
any annotated span of that category.  One class-weighted logistic regression
per category is fitted on the train chunks, ``C`` and a per-category threshold
are picked on validation, and the test set is scored once, at chunk level and
at contract level (a contract is flagged when any of its chunks is).

Run from the project folder::

    python scripts/baseline_tfidf.py                 # 14 shortlisted categories
    python scripts/baseline_tfidf.py --categories all
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

from load_dataset import PROCESSED_DIR, load_annotations, load_contracts

PROJECT_DIR = PROCESSED_DIR.parents[1]
PROFILE_PATH = PROJECT_DIR / "data" / "eda" / "category_profile.csv"
RESULTS_DIR = PROJECT_DIR / "results" / "baseline_tfidf_lr"

TOKENIZER_NAME = "bert-base-uncased"
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64

C_GRID = (0.25, 1.0, 4.0, 16.0, 64.0)
DEFAULT_THRESHOLD = 0.5


def window_bounds(n_tokens: int, size: int = CHUNK_TOKENS,
                  overlap: int = OVERLAP_TOKENS) -> list[tuple[int, int]]:
    """Token ``[start, end)`` of each overlapping window.
    this stops once a window reaches the end, so it never adds a tail window that sits inside the previous one."""
    if n_tokens <= 0:
        return []
    step = size - overlap
    bounds = []
    start = 0
    while True:
        end = min(start + size, n_tokens)
        bounds.append((start, end))
        if end == n_tokens:
            return bounds
        start += step


def chunk_char_bounds(offsets: list[tuple[int, int]], size: int = CHUNK_TOKENS,
                      overlap: int = OVERLAP_TOKENS) -> list[tuple[int, int, int]]:
    """Character ``(start, end, n_tokens)`` of each chunk, from token offsets."""
    return [(offsets[s][0], offsets[e - 1][1], e - s)
            for s, e in window_bounds(len(offsets), size, overlap)]


def load_tokenizer(name: str = TOKENIZER_NAME):
    from transformers import AutoTokenizer  

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.model_max_length = 10**9  
    return tokenizer


def make_chunks(contracts: pd.DataFrame, tokenizer=None) -> pd.DataFrame:
    """One row per chunk: contract_id, chunk_index, start, end, n_tokens, text."""
    tokenizer = tokenizer or load_tokenizer()
    rows = []
    for cid, text in zip(contracts["contract_id"], contracts["text"]):
        offsets = tokenizer(text, add_special_tokens=False,
                            return_offsets_mapping=True)["offset_mapping"]
        for i, (start, end, n) in enumerate(chunk_char_bounds(offsets)):
            rows.append((cid, i, start, end, n, text[start:end]))
    return pd.DataFrame(rows, columns=["contract_id", "chunk_index", "start", "end",
                                       "n_tokens", "text"])


def label_chunks(chunks: pd.DataFrame, annotations: pd.DataFrame,
                 categories: list[str]) -> pd.DataFrame:
    """0/1 matrix (chunks x categories): 1 if the chunk overlaps a span."""
    keep = annotations["category"].isin(categories)
    spans = [
        (cid, cat, a["start"], a["end"])
        for cid, cat, anns in zip(
            annotations.loc[keep, "contract_id"],
            annotations.loc[keep, "category"],
            annotations.loc[keep, "annotations"],
        )
        for a in anns
        if a.get("span_is_valid", True)
    ]
    spans = pd.DataFrame(spans, columns=["contract_id", "category", "s", "e"])

    labels = pd.DataFrame(0, index=chunks.index, columns=categories, dtype=np.int8)
    if spans.empty:
        return labels
    m = chunks[["contract_id", "start", "end"]].reset_index().merge(spans, on="contract_id")
    hit = m[(m["s"] < m["end"]) & (m["e"] > m["start"])]
    for cat, idx in hit.groupby("category")["index"]:
        labels.loc[idx.unique(), cat] = 1
    return labels


def contract_labels(annotations: pd.DataFrame, contract_ids, categories) -> pd.DataFrame:
    """0/1 matrix (contracts x categories) from ``has_annotation``."""
    y = (annotations[annotations["category"].isin(categories)]
         .pivot_table(index="contract_id", columns="category",
                      values="has_annotation", aggfunc="max"))
    return y.reindex(index=list(contract_ids), columns=categories).fillna(0).astype(int)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                           sublinear_tf=True, max_features=200_000,
                           token_pattern=r"(?u)\b\w[\w'-]*\b")


def fit_models(X, Y: pd.DataFrame, C: float) -> dict[str, LogisticRegression]:
    models = {}
    for cat in Y.columns:
        y = Y[cat].to_numpy()
        if y.sum() == 0:
            continue
        models[cat] = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                         max_iter=1000).fit(X, y)
    return models


def predict_proba(models, X, categories) -> pd.DataFrame:
    P = np.zeros((X.shape[0], len(categories)))
    for j, cat in enumerate(categories):
        if cat in models:
            P[:, j] = models[cat].predict_proba(X)[:, 1]
    return pd.DataFrame(P, columns=categories)


def best_f1_threshold(y_true, scores) -> float:
    """Threshold that maximises F1 on validation; 0.5 if there are no positives."""
    if y_true.sum() == 0:
        return DEFAULT_THRESHOLD
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def per_category_metrics(Y: pd.DataFrame, P: pd.DataFrame,
                         thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for cat in Y.columns:
        y = Y[cat].to_numpy()
        p = P[cat].to_numpy()
        pred = p >= thresholds[cat]
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "category": cat,
            "positives": int(y.sum()),
            "positive_rate": y.mean(),
            "threshold": thresholds[cat],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "avg_precision": average_precision_score(y, p) if y.sum() else np.nan,
        })
    return pd.DataFrame(rows).set_index("category")


def summarise(metrics: pd.DataFrame) -> dict[str, float]:
    """Macro and micro averages, skipping categories with no positives (undefined recall)."""
    metrics = metrics.query("positives > 0")
    tp, fp, fn = metrics["tp"].sum(), metrics["fp"].sum(), metrics["fn"].sum()
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "macro_precision": metrics["precision"].mean(),
        "macro_recall": metrics["recall"].mean(),
        "macro_f1": metrics["f1"].mean(),
        "macro_avg_precision": metrics["avg_precision"].mean(),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0,
    }


def top_features(models, vectorizer, n: int = 12) -> pd.DataFrame:
    """Highest-weighted n-grams per category."""
    vocab = np.asarray(vectorizer.get_feature_names_out())
    rows = []
    for cat, model in models.items():
        coef = model.coef_[0]
        top = np.argsort(coef)[::-1][:n]
        rows.append({"category": cat, "top_ngrams": ", ".join(vocab[top])})
    return pd.DataFrame(rows).set_index("category")


def shortlisted_categories(profile_path: Path = PROFILE_PATH) -> list[str]:
    profile = pd.read_csv(profile_path)
    return profile.loc[profile["shortlist"], "category"].tolist()


@dataclass
class BaselineResult:
    categories: list[str]
    C: float
    val_macro_ap_by_C: dict[float, float]
    thresholds: dict[str, float]
    chunk_metrics: pd.DataFrame
    contract_metrics: pd.DataFrame
    summary: pd.DataFrame
    features: pd.DataFrame
    test_predictions: pd.DataFrame
    chunks: pd.DataFrame = field(repr=False)
    models: dict[str, LogisticRegression] = field(repr=False)
    vectorizer: TfidfVectorizer = field(repr=False)


def run_baseline(categories: list[str] | None = None, tokenizer=None,
                 verbose: bool = True) -> BaselineResult:
    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    categories = categories or shortlisted_categories()

    contracts = load_contracts()
    splits = pd.read_csv(PROCESSED_DIR / "splits.csv")
    contracts = (contracts.drop(columns="split")
                 .merge(splits, on="contract_id", how="inner", validate="1:1"))
    annotations = load_annotations()
    annotations = annotations.loc[annotations["contract_id"].isin(contracts["contract_id"])]
    log(f"contracts: {contracts['split'].value_counts().to_dict()}  categories: {len(categories)}")

    chunks = make_chunks(contracts, tokenizer)
    chunks = chunks.merge(contracts[["contract_id", "split"]], on="contract_id")
    Y = label_chunks(chunks, annotations, categories)
    log(f"chunks: {chunks['split'].value_counts().to_dict()}  "
        f"({time.time() - t0:.0f}s)")

    part = {s: (chunks["split"] == s).to_numpy() for s in ("train", "val", "test")}
    vectorizer = make_vectorizer()
    X_train = vectorizer.fit_transform(chunks.loc[part["train"], "text"])
    X_val = vectorizer.transform(chunks.loc[part["val"], "text"])
    X_test = vectorizer.transform(chunks.loc[part["test"], "text"])
    Y_train = Y.take(np.flatnonzero(part["train"])).reset_index(drop=True)
    Y_val = Y.take(np.flatnonzero(part["val"])).reset_index(drop=True)
    Y_test = Y.take(np.flatnonzero(part["test"])).reset_index(drop=True)
    log(f"tf-idf vocabulary: {len(vectorizer.vocabulary_):,} n-grams")

    # C is picked by validation macro average precision, test is never seen here.
    val_ap: dict[float, float] = {}
    for C in C_GRID:
        P_val = predict_proba(fit_models(X_train, Y_train, C), X_val, categories)
        val_ap[C] = float(np.nanmean([average_precision_score(Y_val[c], P_val[c])
                                      for c in categories if Y_val[c].to_numpy().sum() > 0]))
        log(f"  C={C:<5} val macro AP={val_ap[C]:.3f}")
    best_C = max(val_ap, key=val_ap.__getitem__)

    models = fit_models(X_train, Y_train, best_C)
    P_val = predict_proba(models, X_val, categories)
    thresholds = {c: best_f1_threshold(Y_val[c].to_numpy(), P_val[c].to_numpy())
                  for c in categories}

    P_test = predict_proba(models, X_test, categories)
    chunk_metrics = per_category_metrics(Y_test, P_test, thresholds)

    test_chunks = chunks.loc[part["test"], ["contract_id", "chunk_index", "start", "end"]]
    test_chunks = test_chunks.reset_index(drop=True)
    contract_scores = pd.DataFrame(P_test.groupby(test_chunks["contract_id"]).max())
    Yc_test = contract_labels(annotations, contract_scores.index, categories)
    P_contract = pd.DataFrame(contract_scores.to_numpy(), columns=categories)
    contract_metrics = per_category_metrics(Yc_test, P_contract, thresholds)

    summary = pd.DataFrame({"chunk_level": summarise(chunk_metrics),
                            "contract_level": summarise(contract_metrics)})

    predictions = pd.concat([test_chunks,
                             P_test.add_prefix("prob__"),
                             Y_test.add_prefix("label__")], axis=1)
    log(f"done in {time.time() - t0:.0f}s")
    return BaselineResult(categories, best_C, val_ap, thresholds, chunk_metrics,
                          contract_metrics, summary, top_features(models, vectorizer),
                          predictions, chunks, models, vectorizer)


def save_results(result: BaselineResult, out_dir: Path = RESULTS_DIR / "shortlist") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.chunk_metrics.round(4).to_csv(out_dir / "metrics_chunk_level.csv")
    result.contract_metrics.round(4).to_csv(out_dir / "metrics_contract_level.csv")
    result.summary.round(4).to_csv(out_dir / "metrics_summary.csv")
    result.features.to_csv(out_dir / "top_ngrams.csv")
    result.test_predictions.round(6).to_csv(out_dir / "test_predictions.csv", index=False)
    config = {
        "issue": 13,
        "tokenizer": TOKENIZER_NAME,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "chunk_label_rule": "positive if the chunk overlaps any annotated span",
        "vectorizer": {k: v for k, v in make_vectorizer().get_params().items()
                       if k in ("ngram_range", "min_df", "max_df", "sublinear_tf",
                                "max_features", "token_pattern")},
        "model": "LogisticRegression(class_weight='balanced', solver='liblinear'), one per category",
        "C_grid": list(C_GRID),
        "val_macro_avg_precision_by_C": result.val_macro_ap_by_C,
        "chosen_C": result.C,
        "thresholds": "per category, max F1 on validation chunks",
        "contract_rule": "contract flagged if any chunk's probability >= category threshold",
        "categories": result.categories,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=list) + "\n",
                                         encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--categories", choices=["shortlist", "all"], default="shortlist")
    args = parser.parse_args()

    if args.categories == "all":
        categories = json.loads((PROCESSED_DIR / "categories.json").read_text(encoding="utf-8"))
        categories = [c["name"] for c in categories]
        out_dir = RESULTS_DIR / "all_categories"
    else:
        categories, out_dir = None, RESULTS_DIR / "shortlist"

    result = run_baseline(categories)
    save_results(result, out_dir)
    print("\n" + result.summary.round(3).to_string())
    print(f"\nresults written to {out_dir.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
