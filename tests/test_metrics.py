"""Unit tests for offline evaluation metrics: Recall@K, Precision@K, NDCG@K, MAP@K, and Coverage."""

import pytest
import numpy as np
from src.deeprec.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    catalog_coverage,
)


def test_known_answer_precision_and_recall():
    recommended = [10, 20, 30, 40, 50]
    relevant = [20, 40, 60] # 2 hits in top-5 (20 and 40)

    # At k=5: 2 hits out of 5 recommended -> Precision = 2/5 = 0.4
    # 2 hits out of 3 relevant -> Recall = 2/3 = 0.6666...
    assert precision_at_k(recommended, relevant, k=5) == pytest.approx(0.4)
    assert recall_at_k(recommended, relevant, k=5) == pytest.approx(2.0 / 3.0)

    # At k=2: recommended=[10, 20]. 1 hit (20).
    # Precision = 1/2 = 0.5, Recall = 1/3
    assert precision_at_k(recommended, relevant, k=2) == pytest.approx(0.5)
    assert recall_at_k(recommended, relevant, k=2) == pytest.approx(1.0 / 3.0)


def test_known_answer_ndcg():
    # Recommended: [item1, item2, item3]
    # Hits at rank 1 and 3 (0-indexed: 0 and 2)
    recommended = [1, 2, 3]
    relevant = [1, 3]

    # DCG = 1 / log2(1+1) + 0 + 1 / log2(3+1) = 1/1 + 1/2 = 1.5
    # IDCG (ideal: hits at rank 1 and 2) = 1 / log2(2) + 1 / log2(3) = 1.0 + 0.63092975 = 1.63092975
    expected_ndcg = 1.5 / (1.0 + (1.0 / np.log2(3)))

    score = ndcg_at_k(recommended, relevant, k=3)
    assert score == pytest.approx(expected_ndcg, rel=1e-4)

    # Perfect ranking test: recommendations match ideal order
    perfect_rec = [1, 3, 2]
    assert ndcg_at_k(perfect_rec, relevant, k=2) == pytest.approx(1.0)


def test_known_answer_map():
    recommended = [1, 2, 3, 4]
    relevant = [1, 3] # Hits at rank 1 and 3

    # Rank 1: P@1 = 1/1 = 1.0
    # Rank 3: P@3 = 2/3 = 0.6666...
    # AP = (1.0 + 2/3) / 2 = 1.6666... / 2 = 0.8333...
    assert map_at_k(recommended, relevant, k=4) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_empty_inputs():
    assert precision_at_k([], [1, 2], k=5) == 0.0
    assert recall_at_k([1, 2], [], k=5) == 0.0
    assert ndcg_at_k([], [], k=5) == 0.0
    assert map_at_k([], [1], k=5) == 0.0


def test_duplicate_prediction_handling():
    # If list has duplicates, only distinct items are counted towards top-k
    recommended = [10, 10, 20, 20, 30]
    relevant = [20]
    # Unique top-3: [10, 20, 30], hit is 20 at unique rank 2
    assert precision_at_k(recommended, relevant, k=3) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1.0)


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], [1], k=0)

    with pytest.raises(ValueError):
        recall_at_k([1, 2], [1], k=-1)


def test_catalog_coverage():
    all_recs = [
        [1, 2, 3],
        [2, 4, 5],
        [3, 6, 7],
    ]
    total_catalog = 10
    # Unique recommended items: {1, 2, 3, 4, 5, 6, 7} -> 7 items out of 10 = 0.7
    cov = catalog_coverage(all_recs, total_catalog_size=total_catalog)
    assert cov == pytest.approx(0.7)
