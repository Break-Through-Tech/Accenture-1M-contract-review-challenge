"""Load the prepared CUAD files as one table.

The clause view holds ``contract_id``, ``text``, ``category`` and the
annotation fields together.  It is built in memory instead of being saved,
because writing it out would repeat every contract 41 times and produce a
file of roughly 1.1 GB from 26 MB of source text.

    from load_dataset import load_clauses

    clauses = load_clauses(split="train")
    clauses[clauses["has_annotation"]].groupby("category").size()
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run 'python scripts/prepare_cuad.py' first."
        )
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_contracts(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """One row per contract: contract_id, text, split, length metadata."""
    return pd.DataFrame(_read_jsonl(processed_dir / "contracts.jsonl"))


def load_annotations(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """One row per contract and category, with the annotation spans."""
    return pd.DataFrame(_read_jsonl(processed_dir / "annotations.jsonl"))


def load_clauses(
    split: str | None = None,
    labelled_only: bool = False,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Join contracts and annotations into one clause-level table.

    Args:
        split: keep only ``"train"`` or ``"test"``; ``None`` keeps both.
        labelled_only: drop the categories that this contract does not have.
            Roughly 68% of rows are unlabelled, and those negatives are what
            make precision and recall meaningful, so keep them unless a task
            genuinely needs only the positives.
    """
    contracts = load_contracts(processed_dir)
    annotations = load_annotations(processed_dir)

    clauses = annotations.merge(
        contracts[["contract_id", "text"]], on="contract_id", how="left", validate="m:1"
    )
    if clauses["text"].isna().any():
        missing = clauses.loc[clauses["text"].isna(), "contract_id"].unique()
        raise ValueError(f"Annotations reference unknown contracts: {list(missing)[:5]}")

    if split is not None:
        clauses = clauses[clauses["split"] == split]
    if labelled_only:
        clauses = clauses[clauses["has_annotation"]]

    columns = [
        "contract_id",
        "text",
        "category",
        "annotations",
        "has_annotation",
        "split",
    ]
    return clauses[columns].reset_index(drop=True)


if __name__ == "__main__":
    clauses = load_clauses()
    print(f"clause rows: {len(clauses):,}")
    print(f"contracts:   {clauses['contract_id'].nunique()}")
    print(f"categories:  {clauses['category'].nunique()}")
    print(f"labelled:    {int(clauses['has_annotation'].sum()):,}")
    print(clauses[["contract_id", "category", "has_annotation", "split"]].head())
