"""Guard against category loss, data leakage, and dropped boundary evidence."""
import numpy as np
import pandas as pd
import pytest
import torch

from baseline_tfidf import label_chunks, window_bounds
from transformer_multilabel import (ChunkDataset, make_model_chunks,
                                   positive_weights, training_partitions)


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        bounds = window_bounds(len(text), kwargs["max_length"] - 2, kwargs["stride"])
        ids = [[101, *range(s, e), 102] for s, e in bounds]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids],
                "offset_mapping": [[(0, 0), *[(i, i + 1) for i in range(s, e)], (0, 0)]
                                   for s, e in bounds]}


def test_multilabel_chunk_keeps_both_overlapping_categories():
    chunks = make_model_chunks(pd.DataFrame({"contract_id": ["a"], "text": ["abcdefgh"]}),
                               Tokenizer(), max_length=6, overlap=1)
    annotations = pd.DataFrame({"contract_id": ["a", "a"], "category": ["A", "B"],
                                "annotations": [[{"start": 2, "end": 5}],
                                                [{"start": 3, "end": 4}]]})
    y = label_chunks(chunks, annotations, ["A", "B"])
    assert y.iloc[0].tolist() == [1, 1]
    assert y.iloc[1].tolist() == [1, 1]
    sample = ChunkDataset(chunks, y)[0]
    assert sample["labels"].dtype == torch.float32
    assert sample["labels"].shape == (2,)


def test_special_tokens_do_not_truncate_annotated_content():
    chunks = make_model_chunks(pd.DataFrame({"contract_id": ["a"], "text": ["abcdefgh"]}),
                               Tokenizer(), max_length=6, overlap=1)
    assert all(len(ids) <= 6 for ids in chunks["input_ids"])
    covered = {i for ids in chunks["input_ids"] for i in ids[1:-1]}
    assert covered == set(range(8))
    assert chunks[["start", "end"]].values.tolist() == [[0, 4], [3, 7], [6, 8]]


def test_training_weights_use_category_counts_and_handle_absent_labels():
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    np.testing.assert_equal(positive_weights(y, cap=2), [2, 1, 1])
    with pytest.raises(ValueError):
        positive_weights([[2, 0]])


def test_manifest_controls_splits_and_excludes_official_test():
    contracts = pd.DataFrame({"contract_id": ["train", "val", "test", "dup"],
                              "split": ["train", "train", "test", "train"],
                              "is_duplicate_of": [None, None, None, "train"],
                              "text": ["a", "b", "c", "a"]})
    manifest = pd.DataFrame({"contract_id": ["train", "val", "test"],
                             "split": ["train", "val", "test"]})
    parts = training_partitions(contracts, manifest)
    assert parts["train"]["contract_id"].tolist() == ["train"]
    assert parts["val"]["contract_id"].tolist() == ["val"]
    with pytest.raises(ValueError, match="more than once"):
        training_partitions(contracts, pd.concat([manifest, manifest.iloc[:1]]))


def test_weighted_bce_and_sigmoid_allow_two_simultaneous_predictions():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 1.0]])
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0, 2.0]))(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert (logits.grad < 0).all()
    assert (torch.sigmoid(torch.tensor([[2.0, 3.0]])) > 0.5).tolist() == [[True, True]]
