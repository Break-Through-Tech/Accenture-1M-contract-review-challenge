"""Checks that splits.csv is a clean, leak-free split of the prepared contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from load_dataset import load_annotations, load_contracts

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SPLITS_PATH = PROCESSED / "splits.csv"


@pytest.fixture(scope="module")
def splits() -> pd.DataFrame:
    if not SPLITS_PATH.exists():
        pytest.skip("run scripts/make_splits.py first")
    return pd.read_csv(SPLITS_PATH)


@pytest.fixture(scope="module")
def contracts() -> pd.DataFrame:
    return load_contracts()


def ids(df: pd.DataFrame, split: str) -> set[str]:
    return set(df.loc[df["split"] == split, "contract_id"])


@pytest.mark.slow
def test_each_contract_appears_once(splits):
    assert splits["contract_id"].is_unique
    assert set(splits["split"]) == {"train", "val", "test"}


@pytest.mark.slow
def test_splits_do_not_overlap(splits):
    train, val, test = ids(splits, "train"), ids(splits, "val"), ids(splits, "test")
    assert not train & val
    assert not train & test
    assert not val & test


@pytest.mark.slow
def test_official_test_split_is_untouched(splits, contracts):
    unique = contracts[contracts["is_duplicate_of"].isna()]
    assert ids(splits, "test") == ids(unique, "test")


@pytest.mark.slow
def test_val_comes_only_from_official_train(splits, contracts):
    assert ids(splits, "val") <= ids(contracts, "train")


@pytest.mark.slow
def test_duplicates_are_dropped_and_nothing_else_is(splits, contracts):
    duplicates = set(contracts.loc[contracts["is_duplicate_of"].notna(), "contract_id"])
    unique = set(contracts["contract_id"]) - duplicates
    assert not duplicates & set(splits["contract_id"])
    assert set(splits["contract_id"]) == unique


@pytest.mark.slow
def test_val_is_about_a_fifth_of_train(splits):
    n_val = (splits["split"] == "val").sum()
    n_pool = splits["split"].isin(["train", "val"]).sum()
    assert n_val / n_pool == pytest.approx(0.2, abs=0.01)


@pytest.mark.slow
def test_val_positive_rates_match_train(splits):
    """Stratification should keep every common category's rate close in train and val."""
    ann = load_annotations().merge(
        splits.rename(columns={"split": "new_split"}), on="contract_id"
    )
    ann = ann[ann["new_split"].isin(["train", "val"])]
    rates = ann.pivot_table(
        index="category", columns="new_split", values="has_annotation", aggfunc="mean"
    )
    positives = ann.groupby("category")["has_annotation"].sum()

    common = rates[positives >= 50]
    gap = (common["train"] - common["val"]).abs()
    assert gap.max() <= 0.05, gap.sort_values(ascending=False).head()

    val_positives = ann[ann["new_split"] == "val"].groupby("category")["has_annotation"].sum()
    rare_but_present = positives[positives >= 5].index
    assert (val_positives[rare_but_present] >= 1).all()