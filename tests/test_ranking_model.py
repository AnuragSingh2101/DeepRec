"""Unit tests for DeepRankingModel, RankingFeatureExtractor, and ranking inference."""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.deeprec.models.ranking import DeepRankingModel
from src.deeprec.data.ranking_features import RankingFeatureExtractor


def test_ranking_model_forward_and_shapes():
    num_users = 100
    num_movies = 50
    continuous_dim = 27
    batch_size = 8

    model = DeepRankingModel(
        num_users=num_users,
        num_movies=num_movies,
        continuous_dim=continuous_dim,
        embedding_dim=16,
        hidden_dims=[32, 16],
    )
    model.eval()

    u = torch.randint(1, num_users + 1, (batch_size,))
    m = torch.randint(1, num_movies + 1, (batch_size,))
    f = torch.randn(batch_size, continuous_dim)

    logits = model(u, m, f)
    assert logits.shape == (batch_size,)
    assert torch.isfinite(logits).all()


def test_ranking_model_training_and_loss():
    num_users = 20
    num_movies = 30
    continuous_dim = 12

    model = DeepRankingModel(
        num_users=num_users,
        num_movies=num_movies,
        continuous_dim=continuous_dim,
        embedding_dim=8,
        hidden_dims=[16, 8],
    )
    model.train()

    u = torch.tensor([1, 2, 3, 4])
    m = torch.tensor([5, 6, 7, 8])
    f = torch.randn(4, continuous_dim)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])

    criterion = torch.nn.BCEWithLogitsLoss()
    logits = model(u, m, f)
    loss = criterion(logits, y)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Param {name} gradient missing"


def test_ranking_checkpoint_save_and_load(tmp_path):
    num_users = 10
    num_movies = 15
    continuous_dim = 9

    model = DeepRankingModel(
        num_users=num_users,
        num_movies=num_movies,
        continuous_dim=continuous_dim,
        embedding_dim=8,
        hidden_dims=[16, 8],
    )
    model.eval()

    chk_path = tmp_path / "rank_test.pt"
    torch.save({
        "num_users": num_users,
        "num_movies": num_movies,
        "continuous_dim": continuous_dim,
        "embedding_dim": 8,
        "hidden_dims": [16, 8],
        "model_state_dict": model.state_dict(),
    }, chk_path)

    loaded_chk = torch.load(chk_path, map_location="cpu", weights_only=False)
    loaded_model = DeepRankingModel(
        num_users=loaded_chk["num_users"],
        num_movies=loaded_chk["num_movies"],
        continuous_dim=loaded_chk["continuous_dim"],
        embedding_dim=loaded_chk["embedding_dim"],
        hidden_dims=loaded_chk["hidden_dims"],
    )
    loaded_model.load_state_dict(loaded_chk["model_state_dict"])
    loaded_model.eval()

    u = torch.tensor([1, 2])
    m = torch.tensor([3, 4])
    f = torch.randn(2, continuous_dim)

    with torch.no_grad():
        out1 = model(u, m, f)
        out2 = loaded_model(u, m, f)
    torch.testing.assert_close(out1, out2)


def test_ranking_feature_extractor_synthetic():
    train_df = pd.DataFrame({
        "user_idx": [1, 1, 2, 2],
        "movie_idx": [10, 20, 10, 30],
        "rating": [5.0, 3.0, 4.0, 2.0],
        "timestamp": [100, 200, 150, 250],
    })

    movies_df = pd.DataFrame({
        "movie_idx": [10, 20, 30],
        "release_year": [1995, 1998, 2000],
        "genre_Action": [1, 0, 1],
        "genre_Comedy": [0, 1, 1],
    })

    extractor = RankingFeatureExtractor(genres=["Action", "Comedy"])
    extractor.fit(train_df, movies_df)

    # Check fitted values
    assert 10 in extractor.item_popularity
    assert extractor.is_fitted

    # Raw features extraction
    raw_feat = extractor.extract_raw_features(user_idx=1, movie_idx=10, candidate_score=1.5)
    assert len(raw_feat) == 9
    assert np.isfinite(raw_feat).all()

    # Matrix transform with powers
    mat = np.array([raw_feat, raw_feat])
    extractor.fit_scaler(mat)
    transformed = extractor.transform_features(mat)
    # [x, x^2, sqrt(x)] -> 9 * 3 = 27 columns
    assert transformed.shape == (2, 27)
    assert np.isfinite(transformed).all()
