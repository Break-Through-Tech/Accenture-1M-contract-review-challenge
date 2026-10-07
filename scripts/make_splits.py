import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from load_dataset import load_contracts, load_annotations, PROCESSED_DIR

VAL_SIZE = 0.2
SEEDS = range(100)
MIN_POSITIVES = 50  # the test checks rate gaps only for categories this common


def max_rate_gap(Y: pd.DataFrame, val_idx: np.ndarray) -> float:
    is_val = np.zeros(len(Y), dtype=bool)
    is_val[val_idx] = True
    common = Y.columns[Y.sum() >= MIN_POSITIVES]
    gap = (Y.loc[~is_val, common].mean() - Y.loc[is_val, common].mean()).abs()
    return gap.max()


def main() -> None:
    contracts = load_contracts()
    contracts = contracts[contracts["is_duplicate_of"].isna()]
    ann = load_annotations()
    ann = ann[ann["contract_id"].isin(contracts["contract_id"])]

    train_ids = contracts.loc[contracts["split"] == "train", "contract_id"]
    Y = (ann[ann["contract_id"].isin(train_ids)]
         .pivot_table(index="contract_id", columns="category",
                      values="has_annotation", aggfunc="max")
         .astype(int).loc[train_ids])

    best = None
    for seed in SEEDS:
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=seed)  # pyright: ignore[reportArgumentType]
        _, val_idx = next(msss.split(Y.index.to_frame(), Y.values))
        gap = max_rate_gap(Y, val_idx)
        if best is None or gap < best[0]:
            best = (gap, seed, val_idx)

    assert best is not None  # SEEDS is non-empty, so the loop always sets it
    gap, seed, val_idx = best
    print(f"chosen seed: {seed}, max train/val rate gap: {gap:.3f}")

    splits = contracts[["contract_id", "split"]].copy()
    splits.loc[splits["contract_id"].isin(Y.index[val_idx]), "split"] = "val"
    splits.to_csv(PROCESSED_DIR / "splits.csv", index=False)
    print(splits["split"].value_counts())


if __name__ == "__main__":
    main()