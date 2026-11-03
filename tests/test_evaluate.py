import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev
import threshold_experiments as te

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _fake(seed=0, n_contracts=30, per_contract=4, categories=("A", "B")):
    rng = np.random.default_rng(seed)
    contract_ids = np.repeat(np.arange(n_contracts), per_contract)
    Y = (rng.random((len(contract_ids), len(categories))) < 0.3).astype(int)
    P = {"baseline": np.clip(Y * 0.4 + rng.random(Y.shape) * 0.6, 0, 1),
         "transformer": np.clip(Y * 0.6 + rng.random(Y.shape) * 0.4, 0, 1)}
    return list(categories), Y, P, contract_ids


def test_priority_categories_are_cuad_categories():
    config = json.loads((PROJECT_DIR / "results" / "transformer_v1" / "config.json").read_text())
    assert len(ev.PRIORITY) == len(set(ev.PRIORITY)) == 14
    assert set(ev.PRIORITY) <= set(config["categories"])


def test_evaluate_scores_both_models_at_both_levels():
    categories, Y, P, contract_ids = _fake()
    overall, chunk_cat, contract_cat = ev.evaluate(categories, Y, P, contract_ids)

    assert len(overall) == len(ev.MODELS) * 2 * 2  # models x scorings x levels
    assert set(overall["scoring"]) == {"same validation set", "held-out"}
    assert len(chunk_cat) == len(contract_cat) == len(ev.MODELS) * 2 * len(categories)
    assert overall[["micro_f1", "macro_f1"]].apply(lambda s: s.between(0, 1)).all().all()


def test_same_set_scoring_matches_threshold_experiments():
    categories, Y, P, contract_ids = _fake(seed=4)
    overall, _, _ = ev.evaluate(categories, Y, P, contract_ids)
    fit = te.STRATEGIES[ev.STRATEGY]
    expected = te.summarise(Y, P["transformer"] >= fit(Y, P["transformer"]))
    row = overall.query("model == 'transformer' and scoring == 'same validation set' "
                        "and level == 'chunk'").iloc[0]
    assert row["micro_f1"] == expected["micro_f1"]
    assert row["macro_f1"] == expected["macro_f1"]


def test_side_by_side_change_is_transformer_minus_baseline():
    categories, Y, P, contract_ids = _fake(seed=2)
    _, chunk_cat, _ = ev.evaluate(categories, Y, P, contract_ids)
    wide = ev.side_by_side(chunk_cat)
    assert wide["category"].tolist() == categories
    assert np.allclose(wide["change_f1"], wide["transformer_f1"] - wide["baseline_f1"])


def test_load_aligned_rejects_mismatched_labels(tmp_path):
    rows = pd.DataFrame({"contract_id": ["c1", "c1"], "chunk_index": [0, 1],
                         "prob__A": [0.2, 0.8], "label__A": [0, 1]})
    rows.to_csv(tmp_path / "t.csv", index=False)
    rows.assign(label__A=[1, 1]).to_csv(tmp_path / "b.csv", index=False)
    try:
        ev.load_aligned(tmp_path / "t.csv", tmp_path / "b.csv", refresh=False)
    except ValueError as err:
        assert "labels" in str(err)
    else:
        raise AssertionError("expected a label mismatch error")
