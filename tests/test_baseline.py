"""Chunking, chunk labelling and thresholding used by the TF-IDF baseline."""

import numpy as np
import pandas as pd

from baseline_tfidf import (best_f1_threshold, chunk_char_bounds, label_chunks,
                            summarise, window_bounds)


def test_windows_overlap_and_cover_every_token():
    bounds = window_bounds(1000, size=512, overlap=64)
    assert bounds == [(0, 512), (448, 960), (896, 1000)]
    covered = set()
    for start, end in bounds:
        covered.update(range(start, end))
    assert covered == set(range(1000))


def test_no_redundant_tail_window():
    assert window_bounds(512, size=512, overlap=64) == [(0, 512)]
    assert window_bounds(10, size=512, overlap=64) == [(0, 10)]
    assert window_bounds(0) == []


def test_chunk_char_bounds_follow_token_offsets():
    offsets = [(i * 4, i * 4 + 3) for i in range(5)]
    assert chunk_char_bounds(offsets, size=3, overlap=1) == [(0, 11, 3), (8, 19, 3)]


def test_chunk_is_positive_when_it_overlaps_part_of_a_span():
    chunks = pd.DataFrame({"contract_id": ["c1", "c1", "c1", "c2"],
                           "start": [0, 80, 160, 0], "end": [100, 180, 260, 100]})
    span = {"start": 90, "end": 120, "span_is_valid": True}
    annotations = pd.DataFrame({
        "contract_id": ["c1", "c1", "c2"],
        "category": ["Audit Rights", "Insurance", "Audit Rights"],
        "annotations": [[span], [], []],
    })
    labels = label_chunks(chunks, annotations, ["Audit Rights", "Insurance"])
    assert labels["Audit Rights"].tolist() == [1, 1, 0, 0]
    assert labels["Insurance"].tolist() == [0, 0, 0, 0]


def test_touching_but_not_overlapping_is_negative():
    chunks = pd.DataFrame({"contract_id": ["c1"], "start": [0], "end": [100]})
    annotations = pd.DataFrame({"contract_id": ["c1"], "category": ["Insurance"],
                                "annotations": [[{"start": 100, "end": 150}]]})
    assert label_chunks(chunks, annotations, ["Insurance"])["Insurance"].tolist() == [0]


def test_threshold_maximises_f1_and_defaults_without_positives():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert best_f1_threshold(y, scores) == 0.35
    assert best_f1_threshold(np.zeros(3), np.array([0.2, 0.9, 0.5])) == 0.5


def test_summary_skips_categories_without_positives():
    metrics = pd.DataFrame({
        "positives": [10, 0], "tp": [5, 0], "fp": [5, 2], "fn": [5, 0],
        "precision": [0.5, 0.0], "recall": [0.5, 0.0], "f1": [0.5, 0.0],
        "avg_precision": [0.6, np.nan],
    }, index=["A", "B"])
    summary = summarise(metrics)
    assert summary["macro_f1"] == 0.5
    assert summary["micro_precision"] == 0.5
