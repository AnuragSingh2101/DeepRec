"""Unit tests for data preprocessing, ID mapping, and temporal splitting using synthetic fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from src.deeprec.data.preprocess import preprocess_data, GENRE_NAMES


@pytest.fixture
def synthetic_raw_data(tmp_path):
    """Create a minimal synthetic MovieLens directory for isolated testing."""
    raw_dir = tmp_path / "raw" / "ml-100k"
    raw_dir.mkdir(parents=True)

    # 3 users, 6 movies. Users have 6, 7, and 4 interactions respectively.
    # User 3 has only 4 interactions (<5) so should be filtered out.
    ratings_data = [
        # user_id, movie_id, rating, timestamp
        # User 1: 6 interactions
        (1, 101, 5, 1000),
        (1, 102, 4, 1010),
        (1, 103, 3, 1020),
        (1, 104, 5, 1030),
        (1, 105, 4, 1040),
        (1, 106, 5, 1050),
        # User 2: 7 interactions
        (2, 102, 2, 2000),
        (2, 101, 5, 2010),
        (2, 104, 4, 2020),
        (2, 103, 3, 2030),
        (2, 106, 5, 2040),
        (2, 105, 4, 2050),
        (2, 107, 5, 2060),
        # User 3: 4 interactions (below threshold 5)
        (3, 101, 4, 3000),
        (3, 102, 3, 3010),
        (3, 103, 5, 3020),
        (3, 104, 2, 3030),
    ]

    ratings_df = pd.DataFrame(ratings_data, columns=["user_id", "movie_id", "rating", "timestamp"])
    ratings_df.to_csv(raw_dir / "u.data", sep="\t", header=False, index=False)

    # Synthetic movies catalogue (u.item)
    # 7 movies: 101 to 107
    movies_rows = []
    for mid in range(101, 108):
        row = [mid, f"Movie {mid}", "01-Jan-1995", "", f"http://imdb.com/{mid}"] + [1 if i == 1 else 0 for i in range(19)]
        movies_rows.append(row)
    movies_df = pd.DataFrame(movies_rows)
    movies_df.to_csv(raw_dir / "u.item", sep="|", header=False, index=False)

    # Synthetic users (u.user)
    users_rows = [
        [1, 24, "M", "engineer", "94043"],
        [2, 35, "F", "educator", "10001"],
        [3, 18, "M", "student", "55414"],
    ]
    users_df = pd.DataFrame(users_rows)
    users_df.to_csv(raw_dir / "u.user", sep="|", header=False, index=False)

    out_dir = tmp_path / "processed"
    return raw_dir, out_dir


def test_preprocess_filtering_and_temporal_split(synthetic_raw_data):
    raw_dir, out_dir = synthetic_raw_data
    meta = preprocess_data(raw_dir=raw_dir, out_dir=out_dir, min_interactions=5)

    # Check user filtering: user 3 had only 4 interactions and should be filtered out
    assert meta["num_users"] == 2
    assert meta["num_movies"] == 7

    # Check files exist
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "validation.csv").exists()
    assert (out_dir / "test.csv").exists()
    assert (out_dir / "movies.csv").exists()
    assert (out_dir / "metadata.json").exists()

    train_df = pd.read_csv(out_dir / "train.csv")
    val_df = pd.read_csv(out_dir / "validation.csv")
    test_df = pd.read_csv(out_dir / "test.csv")

    # Each valid user has exactly 1 val target and 1 test target
    assert len(val_df) == 2
    assert len(test_df) == 2

    # User 1 had 6 interactions -> 4 train, 1 val, 1 test
    u1_train = train_df[train_df["user_idx"] == 1]
    assert len(u1_train) == 4

    # User 2 had 7 interactions -> 5 train, 1 val, 1 test
    u2_train = train_df[train_df["user_idx"] == 2]
    assert len(u2_train) == 5

    # Check temporal ordering: for each user, max(train timestamp) < val timestamp < test timestamp
    for u in [1, 2]:
        u_train_ts = train_df[train_df["user_idx"] == u]["timestamp"].max()
        u_val_ts = val_df[val_df["user_idx"] == u]["timestamp"].iloc[0]
        u_test_ts = test_df[test_df["user_idx"] == u]["timestamp"].iloc[0]

        assert u_train_ts < u_val_ts, f"User {u}: train ts {u_train_ts} >= val ts {u_val_ts}"
        assert u_val_ts < u_test_ts, f"User {u}: val ts {u_val_ts} >= test ts {u_test_ts}"


def test_zero_padding_reservation(synthetic_raw_data):
    raw_dir, out_dir = synthetic_raw_data
    preprocess_data(raw_dir=raw_dir, out_dir=out_dir, min_interactions=5)

    with open(out_dir / "movie_map.json") as f:
        movie_map = json.load(f)

    # Verify ID 0 is not assigned to any movie
    assigned_indices = list(movie_map.values())
    assert 0 not in assigned_indices
    assert min(assigned_indices) == 1
    assert max(assigned_indices) == 7


def test_deterministic_mapping(synthetic_raw_data):
    raw_dir, out_dir = synthetic_raw_data
    preprocess_data(raw_dir=raw_dir, out_dir=out_dir, min_interactions=5)

    with open(out_dir / "movie_map.json") as f:
        map1 = json.load(f)

    # Run again on fresh output dir
    out_dir2 = out_dir.parent / "processed2"
    preprocess_data(raw_dir=raw_dir, out_dir=out_dir2, min_interactions=5)

    with open(out_dir2 / "movie_map.json") as f:
        map2 = json.load(f)

    assert map1 == map2, "Mapping is not deterministic across runs!"
