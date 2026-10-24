"""Unit tests for FastAPI endpoints using TestClient."""

import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient
import torch

from src.deeprec.api.main import app
import src.deeprec.api.main as api_main


@pytest.fixture
def client():
    """Create a test client with a mocked RecommendationService."""
    mock_svc = MagicMock()
    mock_svc.is_loaded = True
    mock_svc.device = torch.device("cpu")
    mock_svc.candidate_model = MagicMock()
    mock_svc.ranking_model = MagicMock()
    mock_svc.user_map = {1: 1, 2: 2}
    mock_svc.movie_catalogue = {
        1: {
            "movie_id": 101,
            "movie_idx": 1,
            "title": "Star Wars (1977)",
            "genres": ["Action", "Sci-Fi"],
            "release_year": 1977,
        },
        2: {
            "movie_id": 102,
            "movie_idx": 2,
            "title": "Toy Story (1995)",
            "genres": ["Animation", "Children's"],
            "release_year": 1995,
        },
    }

    mock_svc.get_recommendations.return_value = {
        "user_id": 1,
        "user_idx": 1,
        "is_cold_start": False,
        "candidate_count": 100,
        "returned_count": 2,
        "history": [{"movie_id": 101, "title": "Star Wars (1977)", "genres": ["Action"]}],
        "recommendations": [
            {
                "rank": 1,
                "movie_id": 102,
                "movie_idx": 2,
                "title": "Toy Story (1995)",
                "genres": ["Animation"],
                "release_year": 1995,
                "candidate_score": 2.45,
                "ranking_score": 0.88,
                "explanation": "High engagement candidate",
            }
        ],
    }

    mock_svc.record_interaction.return_value = {
        "status": "recorded",
        "message": "Recorded watch for user 1 and movie 102",
        "interaction": {"user_id": 1, "movie_id": 102, "interaction_type": "watch"},
        "total_user_interactions": 5,
    }

    with TestClient(app) as test_client:
        api_main.service = mock_svc
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["artifacts_loaded"] is True
    assert data["candidate_model_loaded"] is True


def test_movies_endpoint(client):
    response = client.get("/movies?limit=5")
    assert response.status_code == 200
    movies = response.json()
    assert len(movies) == 2
    assert movies[0]["title"] == "Star Wars (1977)"
    assert movies[0]["movie_id"] == 101


def test_recommendations_endpoint(client):
    response = client.get("/recommendations/1?candidate_k=50&top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["movie_id"] == 102


def test_recommendations_invalid_user(client):
    response = client.get("/recommendations/-1")
    assert response.status_code in (400, 422)


def test_record_interaction_endpoint(client):
    payload = {
        "user_id": 1,
        "movie_id": 102,
        "interaction_type": "rating",
        "rating": 5.0,
    }
    response = client.post("/interactions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"


def test_record_interaction_invalid_rating(client):
    payload = {
        "user_id": 1,
        "movie_id": 102,
        "interaction_type": "rating",
        "rating": 6.5, # Out of range (1.0 to 5.0)
    }
    response = client.post("/interactions", json=payload)
    assert response.status_code == 422
