"""Unit tests for sequence generation, padding, masking, and target leakage prevention."""

import pytest
import numpy as np
import pandas as pd

from src.deeprec.data.sequences import build_rolling_sequences, build_eval_sequences


def test_no_target_leakage():
    """Ensure the target item never appears in the input history for that sample."""
    interactions = pd.DataFrame({
        "user_idx": [1, 1, 1, 1, 1],
        "movie_idx": [10, 20, 30, 40, 50],
        "timestamp": [100, 200, 300, 400, 500]
    })

    histories, targets, users = build_rolling_sequences(
        interactions,
        max_history_length=4,
        min_history=1,
        padding_idx=0
    )

    # 5 interactions -> 4 sequence examples:
    # 1. history: [0, 0, 0, 10], target: 20
    # 2. history: [0, 0, 10, 20], target: 30
    # 3. history: [0, 10, 20, 30], target: 40
    # 4. history: [10, 20, 30, 40], target: 50
    assert len(histories) == 4
    assert len(targets) == 4

    for i in range(len(histories)):
        hist = histories[i]
        tgt = targets[i]
        assert tgt not in hist, f"Leakage detected: target {tgt} found in input history {hist}"


def test_left_padding_and_truncation():
    """Test that short sequences are correctly padded with zeros and long ones are truncated."""
    interactions = pd.DataFrame({
        "user_idx": [1, 1, 1, 1],
        "movie_idx": [5, 6, 7, 8],
        "timestamp": [1, 2, 3, 4]
    })

    # max_history_length = 2
    histories, targets, _ = build_rolling_sequences(
        interactions,
        max_history_length=2,
        min_history=1,
        padding_idx=0
    )

    # Step 1: hist: [0, 5], tgt: 6
    np.testing.assert_array_equal(histories[0], [0, 5])
    assert targets[0] == 6

    # Step 2: hist: [5, 6], tgt: 7
    np.testing.assert_array_equal(histories[1], [5, 6])
    assert targets[1] == 7

    # Step 3: hist was [5, 6, 7], truncated to [6, 7], tgt: 8
    np.testing.assert_array_equal(histories[2], [6, 7])
    assert targets[2] == 8


def test_eval_sequences_temporal_integrity():
    """Verify that validation and test sequences strictly use past interactions."""
    train_df = pd.DataFrame({
        "user_idx": [1, 1, 2, 2],
        "movie_idx": [11, 12, 21, 22],
        "timestamp": [10, 20, 15, 25]
    })

    val_df = pd.DataFrame({
        "user_idx": [1, 2],
        "movie_idx": [13, 23],
        "timestamp": [30, 35]
    })

    test_df = pd.DataFrame({
        "user_idx": [1, 2],
        "movie_idx": [14, 24],
        "timestamp": [40, 45]
    })

    # Validation sequences: history should be train items only
    val_hist, val_tgt, val_u = build_eval_sequences(train_df, val_df, max_history_length=3, padding_idx=0)
    assert len(val_hist) == 2

    # User 1 val: hist [0, 11, 12], target 13
    np.testing.assert_array_equal(val_hist[0], [0, 11, 12])
    assert val_tgt[0] == 13

    # User 2 val: hist [0, 21, 22], target 23
    np.testing.assert_array_equal(val_hist[1], [0, 21, 22])
    assert val_tgt[1] == 23

    # Test sequences: history should include train + val
    test_hist, test_tgt, test_u = build_eval_sequences(
        train_df, test_df, max_history_length=3, padding_idx=0, prior_target_df=val_df
    )
    # User 1 test: train=[11, 12] + val=[13] => [11, 12, 13], target 14
    np.testing.assert_array_equal(test_hist[0], [11, 12, 13])
    assert test_tgt[0] == 14

    # User 2 test: train=[21, 22] + val=[23] => [21, 22, 23], target 24
    np.testing.assert_array_equal(test_hist[1], [21, 22, 23])
    assert test_tgt[1] == 24
