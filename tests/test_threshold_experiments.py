import numpy as np
import pytest

import threshold_experiments as te


def test_fit_max_f1_separates_clean_scores():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = te.fit_max_f1(y, p)
    assert 0.3 < threshold <= 0.7


def test_fit_rules_fall_back_to_default_without_positives():
    y = np.zeros(4, dtype=int)
    p = np.array([0.1, 0.4, 0.6, 0.9])
    assert te.fit_max_f1(y, p) == te.DEFAULT_THRESHOLD
    assert te.fit_recall_target(y, p) == te.DEFAULT_THRESHOLD


def test_recall_target_picks_highest_threshold_that_reaches_it():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2])
    threshold = te.fit_recall_target(y, p, target=0.75)
    pred = p >= threshold
    assert (pred & (y == 1)).sum() / y.sum() >= 0.75
    assert threshold == pytest.approx(0.7)


def test_summarise_micro_and_macro_skip_categories_without_positives():
    Y = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    pred = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    metrics = te.summarise(Y, pred)
    assert metrics["micro_precision"] == pytest.approx(0.5)
    assert metrics["micro_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)  # second category ignored


def test_contract_level_flags_contract_if_any_chunk_is_positive():
    codes = np.array([0, 0, 1, 1])
    values = np.array([[0], [1], [0], [0]])
    out = te.any_per_contract(values, codes, 2)
    assert out.ravel().tolist() == [1, 0]


def test_cross_validation_scores_each_contract_with_thresholds_from_other_contracts():
    rng = np.random.default_rng(0)
    n_contracts, per_contract = 40, 3
    contract_ids = np.repeat(np.arange(n_contracts), per_contract)
    Y = (rng.random((len(contract_ids), 2)) < 0.3).astype(int)
    P = np.clip(Y * 0.6 + rng.random(Y.shape) * 0.4, 0, 1)
    codes, n = te.contract_codes(contract_ids)

    seen = []

    def spy(Y_fit, P_fit):
        seen.append(len(Y_fit))
        return np.full(Y_fit.shape[1], 0.5)

    pred = te.cross_validated_predictions(spy, Y, P, codes, n, seed=1)
    assert pred.shape == Y.shape
    assert len(seen) == te.N_FOLDS
    assert all(size < len(Y) for size in seen)
    assert np.array_equal(pred, P >= 0.5)


def test_run_experiments_output_shape(tmp_path):
    rng = np.random.default_rng(3)
    contract_ids = np.repeat(np.arange(30), 4)
    categories = ["A", "B"]
    Y = (rng.random((120, 2)) < 0.25).astype(int)
    P = np.clip(Y * 0.5 + rng.random(Y.shape) * 0.5, 0, 1)
    table, thresholds, per_category = te.run_experiments(categories, Y, P, contract_ids)

    assert set(table["level"]) == {"chunk", "contract"}
    assert set(table["strategy"]) == set(te.STRATEGIES)
    assert list(thresholds["category"]) == categories
    assert len(per_category) == len(te.STRATEGIES) * len(categories)
    held_out = table[table["scoring"].str.contains("held-out")]
    assert not held_out["micro_f1_std"].isna().any()
    assert "fixed_0.5" not in set(held_out["strategy"])
