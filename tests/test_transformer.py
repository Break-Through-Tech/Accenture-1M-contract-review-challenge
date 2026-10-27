"""Example building, class weights and batching used by the transformer."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from baseline_tfidf import chunk_char_bounds  # noqa: E402
from train_transformer import (MAX_CONTENT_TOKENS, collate, epoch_sample,  # noqa: E402
                               make_examples, positive_weights)


class WordTokenizer:
    """One token per space-separated word, with exact character offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offsets, pos = [], 0
        for word in text.split(" "):
            offsets.append((pos, pos + len(word)))
            pos += len(word) + 1
        return {"input_ids": list(range(100, 100 + len(offsets))), "offset_mapping": offsets}


def test_examples_use_baseline_chunks_and_fit_bert():
    text = " ".join(["w"] * 1000)
    contracts = pd.DataFrame({"contract_id": ["c1"], "text": [text], "split": ["train"]})
    chunks, ids = make_examples(contracts, WordTokenizer())
    offsets = WordTokenizer()(text)["offset_mapping"]
    expected = chunk_char_bounds(offsets)
    assert list(zip(chunks["start"], chunks["end"], chunks["n_tokens"])) == expected
    assert [len(x) for x in ids] == [MAX_CONTENT_TOKENS, MAX_CONTENT_TOKENS, 104]
    assert ids[1][0] == 100 + 448  # second window starts 64 tokens before the first one ends
    assert chunks["split"].tolist() == ["train"] * 3


def test_positive_weights_soften_the_ratio_and_skip_empty_categories():
    Y = np.array([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    assert positive_weights(Y).tolist() == [2.0, 1.0]  # sqrt(4 / 1); no positives -> 1


def test_epoch_sample_keeps_every_clause_chunk_and_a_share_of_empty_ones():
    train_idx = np.arange(100, 120)
    has_clause = np.array([True] * 4 + [False] * 16)
    rows = epoch_sample(train_idx, has_clause, 0.25, np.random.default_rng(0))
    assert set(range(100, 104)) <= set(rows)          # all 4 clause chunks
    assert len(rows) == 4 + 4 and len(set(rows)) == 8  # plus a quarter of 16 empty, no repeats
    assert set(rows) <= set(train_idx)


def test_collate_adds_special_tokens_and_pads():
    batch = [([5, 6, 7], np.array([1, 0], dtype=np.float32)),
             ([8], np.array([0, 1], dtype=np.float32))]
    out = collate(batch, cls_id=101, sep_id=102, pad_id=0)
    assert out["input_ids"].tolist() == [[101, 5, 6, 7, 102], [101, 8, 102, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert out["labels"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
