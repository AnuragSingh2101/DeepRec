"""Unit tests for RecommendationService orchestrator, filtering, cold start, and schemas."""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.deeprec.service.recommender import RecommendationService
from src.deeprec.models.candidate_generation import CandidateGenerator
from src.deeprec.models.ranking import DeepRankingModel
from src.deeprec.retrieval.retrieve import CandidateRetriever


@pytest.fixture
def mock_service():
    """Create a RecommendationService instance with mocked components for fast isolated testing."""
    with patch.object(RecommendationService, "load_artifacts", return_value=True):
        svc = RecommendationService()
        svc.device = torch.device("cpu")
        svc.is_loaded = True

        # Mock catalogue: 10 movies
        svc.movie_catalogue = {
            i: {
                "movie_idx": i,
                "movie_id": 100 + i,
                "title": f"Test Movie {i}",
                "genres": ["Action" if i % 2 == 0 else "Drama"],
                "release_year": 1995 + i,
            }
            for i in range(1, 11)
        }
        svc.movie_map = {100 + i: i for i in range(1, 11)}
        svc.idx_to_movie = {i: 100 + i for i in range(1, 11)}

        # Mock users: user 1 has watched movies [1, 2, 3]
        svc.user_map = {1: 1, 2: 2}
        svc.idx_to_user = {1: 1, 2: 2}
        svc.user_histories = {1: [1, 2, 3], 2: [4, 5]}
        svc.user_seen_items = {1: {1, 2, 3}, 2: {4, 5}}
        svc.popular_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Mock candidate retriever
        cand_model = CandidateGenerator(num_items=10, embedding_dim=8, hidden_dims=[16, 8], use_context=False)
        svc.candidate_model = cand_model
        svc.retriever = CandidateRetriever(cand_model, 10, torch.device("cpu"), svc.popular_items)

        # Mock ranking model
        rank_model = DeepRankingModel(num_users=2, num_movies=10, continuous_dim=9, embedding_dim=8, hidden_dims=[16, 8])
        svc.ranking_model = rank_model

        # Mock feature extractor
        mock_fe = MagicMock()
        mock_fe.extract_raw_features.return_value = np.zeros(9, dtype=np.float32)
        mock_fe.transform_features.side_effect = lambda x: np.zeros((len(x), 9), dtype=np.float32)
        mock_fe.user_genre_profiles = {1: np.array([1.0, 0.0])}
        mock_fe.movie_genre_vectors = {i: np.array([1.0, 0.0]) for i in range(1, 11)}
        svc.feature_extractor = mock_fe

        return svc


def test_recommender_seen_item_filtering(mock_service):
    # User 1 has seen movies 1, 2, 3
    res = mock_service.get_recommendations(user_id=1, candidate_k=8, top_n=5, exclude_seen=True)

    assert res["user_id"] == 1
    assert not res["is_cold_start"]
    assert len(res["recommendations"]) <= 5

    recommended_indices = [r["movie_idx"] for r in res["recommendations"]]
    # None of the recommended items should be in seen items [1, 2, 3]
    for seen_id in [1, 2, 3]:
        assert seen_id not in recommended_indices, f"Seen item {seen_id} was recommended!"


def test_recommender_cold_start_fallback(mock_service):
    # User 999 is unknown (cold start)
    res = mock_service.get_recommendations(user_id=999, candidate_k=10, top_n=4)

    assert res["user_id"] == 999
    assert res["is_cold_start"] is True
    assert len(res["recommendations"]) == 4
    # Cold start should recommend popular items
    top_pop_indices = mock_service.popular_items[:4]
    rec_indices = [r["movie_idx"] for r in res["recommendations"]]
    assert rec_indices == top_pop_indices


def test_recommender_record_interaction(mock_service):
    # Add new interaction for user 1 on movie 107 (idx 7)
    res = mock_service.record_interaction(user_id=1, movie_id=107, interaction_type="watch")

    assert res["status"] == "recorded"
    assert 7 in mock_service.user_seen_items[1]
    assert mock_service.user_histories[1][-1] == 7

    # When requesting recommendations with exclude_seen=True, movie 7 must now be excluded
    new_recs = mock_service.get_recommendations(user_id=1, candidate_k=8, top_n=5, exclude_seen=True)
    rec_indices = [r["movie_idx"] for r in new_recs["recommendations"]]
    assert 7 not in rec_indices


def test_recommender_schema_stability(mock_service):
    res = mock_service.get_recommendations(user_id=2, candidate_k=5, top_n=3)

    assert "user_id" in res
    assert "user_idx" in res
    assert "is_cold_start" in res
    assert "recommendations" in res
    assert "history" in res

    first_rec = res["recommendations"][0]
    expected_fields = ["rank", "movie_id", "movie_idx", "title", "genres", "release_year", "candidate_score", "ranking_score", "explanation"]
    for field in expected_fields:
        assert field in first_rec, f"Missing field {field} in recommendation schema"
