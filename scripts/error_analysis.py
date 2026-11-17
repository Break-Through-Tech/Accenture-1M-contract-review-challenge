"""Error analysis of transformer v1 vs the TF-IDF baseline on validation chunks.

Uses the saved validation predictions (no retraining) and the annotated spans. Thresholds
are per-category max F1 with the rare-category fallback, fitted on validation, so counts
are slightly optimistic for both models. Test split unused.

    python scripts/error_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev
import threshold_experiments as te

OUT = ev.OUTPUT_DIR.parent / "error_analysis"
WEAK = ["IP Ownership Assignment", "Termination for Convenience", "Exclusivity", "Minimum Commitment"]
PROC = ev.PROJECT_DIR / "data" / "processed"


def span_coverage(frame, categories, Y):
    """For each positive (chunk, category): 'full' if some annotated span lies wholly inside
    the chunk, else 'partial' (the span runs past a chunk edge). Also span length in chars."""
    ann = pd.read_json(PROC / "annotations.jsonl", lines=True)
    spans = {(r.contract_id, r.category): [a for a in r.annotations if a.get("span_is_valid", True)]
             for r in ann.itertuples()}
    cov = np.full(Y.shape, "", dtype=object)
    length = np.full(Y.shape, np.nan)
    for i, row in enumerate(frame.itertuples()):
        for j, c in enumerate(categories):
            if not Y[i, j]:
                continue
            ov = [a for a in spans.get((row.contract_id, c), []) if a["start"] < row.end and a["end"] > row.start]
            if not ov:
                continue
            full = [a for a in ov if a["start"] >= row.start and a["end"] <= row.end]
            cov[i, j] = "full" if full else "partial"
            length[i, j] = max((a["end"] - a["start"]) for a in (full or ov))
    return cov, length, spans


def main() -> None:
    cats, Y, P, ids = ev.load_aligned(ev.TRANSFORMER_PREDICTIONS, ev.BASELINE_PREDICTIONS, False)
    frame = pd.read_csv(ev.TRANSFORMER_PREDICTIONS).sort_values(ev.KEY).reset_index(drop=True)
    fit = te.STRATEGIES[ev.STRATEGY]
    pred = {m: P[m] >= fit(Y, P[m]) for m in ev.MODELS}
    cov, length, spans = span_coverage(frame, cats, Y)
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Confusion counts per model for the weak priority categories
    rows = []
    for c in WEAK:
        j = cats.index(c)
        for m in ev.MODELS:
            tp = int((pred[m][:, j] & (Y[:, j] == 1)).sum())
            fp = int((pred[m][:, j] & (Y[:, j] == 0)).sum())
            fn = int((~pred[m][:, j] & (Y[:, j] == 1)).sum())
            rows.append({"category": c, "model": m, "positives": int(Y[:, j].sum()), "TP": tp, "FP": fp, "FN": fn})
    pd.DataFrame(rows).to_csv(OUT / "weak_categories_confusion.csv", index=False)

    # 2. Does a span split across a chunk edge explain misses? (all categories pooled)
    rows = []
    for m in ev.MODELS:
        for kind in ("full", "partial"):
            mask = (cov == kind)
            n = int(mask.sum())
            missed = int((mask & ~pred[m]).sum())
            rows.append({"model": m, "span_in_chunk": kind, "positive_chunks": n, "missed": missed,
                         "miss_rate": round(missed / n, 3) if n else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "miss_rate_by_span_coverage.csv", index=False)

    # 3. Miss rate by annotated span length (full-span positives only)
    bins = [0, 100, 300, 1000, 10**9]
    labels = ["<100 chars", "100-300", "300-1000", ">1000"]
    rows = []
    mask_full = (cov == "full")
    for m in ev.MODELS:
        b = pd.cut(pd.Series(length[mask_full]), bins, labels=labels)
        missed = pd.Series((~pred[m])[mask_full])
        t = pd.DataFrame({"bin": b.to_numpy(), "missed": missed.to_numpy()}).groupby("bin", observed=True)["missed"].agg(["size", "sum"])
        for k, r in t.iterrows():
            rows.append({"model": m, "span_length": k, "positive_chunks": int(r["size"]),
                         "missed": int(r["sum"]), "miss_rate": round(r["sum"] / r["size"], 3)})
    pd.DataFrame(rows).to_csv(OUT / "miss_rate_by_span_length.csv", index=False)

    # 4. Disagreements on the weak categories: transformer misses what the baseline gets
    text = {r.contract_id: r.text for r in pd.read_json(PROC / "contracts.jsonl", lines=True).itertuples()}
    ex = []
    for c in WEAK:
        j = cats.index(c)
        t_miss = np.where((Y[:, j] == 1) & ~pred["transformer"][:, j] & pred["baseline"][:, j])[0]
        t_fp = np.where((Y[:, j] == 0) & pred["transformer"][:, j] & ~pred["baseline"][:, j])[0]
        for kind, idx in (("transformer missed, baseline caught", t_miss), ("transformer false alarm, baseline clean", t_fp)):
            for i in idx[:3]:
                r = frame.iloc[i]
                snip = text[r.contract_id][int(r.start):int(r.end)]
                if kind.startswith("transformer missed"):
                    sp = [a for a in spans.get((r.contract_id, c), []) if a["start"] < r.end and a["end"] > r.start]
                    snip = sp[0]["text"] if sp else snip
                ex.append({"category": c, "case": kind, "contract": r.contract_id[:40], "chunk": int(r.chunk_index),
                           "p_transformer": round(float(P["transformer"][i, j]), 3),
                           "p_baseline": round(float(P["baseline"][i, j]), 3),
                           "text": " ".join(snip.split())[:260]})
        ex.append({"category": c, "case": "counts", "contract": "", "chunk": -1,
                   "p_transformer": len(t_miss), "p_baseline": len(t_fp),
                   "text": f"{len(t_miss)} chunks missed only by transformer; {len(t_fp)} false alarms only by transformer"})
    pd.DataFrame(ex).to_csv(OUT / "examples.csv", index=False)
    for f in sorted(OUT.glob("*.csv")):
        print("\n==", f.name); print(pd.read_csv(f).drop(columns=["text"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
