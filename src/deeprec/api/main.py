"""FastAPI web service exposing DeepRec recommendations and metrics."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.deeprec.service.recommender import RecommendationService
from src.deeprec.config import load_config


# Global singleton service
service: Optional[RecommendationService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and metadata once at application startup."""
    global service
    print("[DeepRec API] Initializing RecommendationService...")
    try:
        service = RecommendationService()
        if service.is_loaded:
            print("[DeepRec API] Models and metadata loaded successfully.")
        else:
            print("[DeepRec API] Notice: Model artifacts not found yet. Train models to enable full inference.")
    except Exception as e:
        print(f"[DeepRec API] Startup initialization note: {e}")
        service = None
    yield
    print("[DeepRec API] Shutting down.")


app = FastAPI(
    title="DeepRec Recommendation API",
    description="Two-Stage Deep Learning Video Recommendation System inspired by Covington et al. (YouTube 2016).",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic Schemas ---

class MovieItem(BaseModel):
    movie_id: int
    movie_idx: int
    title: str
    genres: List[str]
    release_year: int


class RecommendationItem(BaseModel):
    rank: int
    movie_id: int
    movie_idx: int
    title: str
    genres: List[str]
    release_year: int
    candidate_score: float
    ranking_score: float
    explanation: str


class RecommendationResponse(BaseModel):
    user_id: int
    user_idx: int
    is_cold_start: bool
    candidate_count: int
    returned_count: int
    history: List[Dict[str, Any]]
    recommendations: List[RecommendationItem]


class InteractionRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="Raw user ID (must be > 0)")
    movie_id: int = Field(..., gt=0, description="Raw movie ID (must be > 0)")
    interaction_type: str = Field(default="watch", description="'watch', 'like', or 'rating'")
    rating: Optional[float] = Field(default=None, ge=1.0, le=5.0, description="Rating between 1.0 and 5.0")


class InteractionResponse(BaseModel):
    status: str
    message: str
    interaction: Dict[str, Any]
    total_user_interactions: int


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool
    candidate_model_loaded: bool
    ranking_model_loaded: bool
    total_catalogue_movies: int
    total_users_indexed: int
    device: str


from fastapi.responses import RedirectResponse


# --- Endpoints ---

@app.get("/", include_in_schema=False)
def root():
    """Redirect root path directly to interactive API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns application status and indicates whether model artifacts are loaded."""
    if service is None or not service.is_loaded:
        return HealthResponse(
            status="degraded (artifacts not loaded)",
            artifacts_loaded=False,
            candidate_model_loaded=False,
            ranking_model_loaded=False,
            total_catalogue_movies=0,
            total_users_indexed=0,
            device="none",
        )

    return HealthResponse(
        status="healthy",
        artifacts_loaded=service.is_loaded,
        candidate_model_loaded=service.candidate_model is not None,
        ranking_model_loaded=service.ranking_model is not None,
        total_catalogue_movies=len(service.movie_catalogue),
        total_users_indexed=len(service.user_map),
        device=str(service.device),
    )


@app.get("/movies", response_model=List[MovieItem])
def list_movies(
    limit: int = Query(default=20, ge=1, le=100, description="Number of movies to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    genre: Optional[str] = Query(default=None, description="Optional genre filter"),
):
    """Returns movie catalogue items with pagination and optional genre filtering."""
    if service is None or not service.movie_catalogue:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Movie catalogue is not loaded. Preprocess the dataset first.",
        )

    items = list(service.movie_catalogue.values())
    if genre:
        items = [m for m in items if genre.lower() in [g.lower() for g in m.get("genres", [])]]

    paginated = items[offset: offset + limit]
    return paginated


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_user_recommendations(
    user_id: int,
    candidate_k: int = Query(default=100, ge=10, le=500, description="Number of Stage 1 candidates"),
    top_n: int = Query(default=10, ge=1, le=50, description="Number of final recommendations"),
    exclude_seen: bool = Query(default=True, description="Exclude items already in user history"),
):
    """Retrieve two-stage personalized recommendations for a user.
    
    If the user has no interaction history, a graceful popularity-based cold-start fallback is returned.
    """
    if user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user_id {user_id}. Must be a positive integer.",
        )

    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation artifacts not loaded. Run 'python -m src.deeprec.training.train_candidates' and 'train_ranker' first.",
        )

    try:
        recs = service.get_recommendations(
            user_id=user_id,
            candidate_k=candidate_k,
            top_n=top_n,
            exclude_seen=exclude_seen,
        )
        return recs
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}",
        )


@app.post("/interactions", response_model=InteractionResponse)
def record_interaction(interaction: InteractionRequest):
    """Record a user interaction in the lightweight in-memory buffer.
    
    Subsequent recommendation calls for this user immediately account for this new interaction.
    Note: In-memory store resets when the server is restarted.
    """
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service not loaded.",
        )

    try:
        result = service.record_interaction(
            user_id=interaction.user_id,
            movie_id=interaction.movie_id,
            interaction_type=interaction.interaction_type,
            rating=interaction.rating,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/metrics")
def get_offline_metrics():
    """Returns saved offline evaluation results and experiment metrics.
    
    Metrics are read strictly from saved artifacts without fabricating results.
    """
    cfg = load_config()
    eval_dir = Path(cfg["paths"]["evaluation_artifact_dir"])
    summary_file = eval_dir / "evaluation_summary.json"
    experiments_file = eval_dir / "experiment_results.csv"

    response = {}
    if summary_file.exists():
        with open(summary_file, "r") as f:
            response["evaluation_summary"] = json.load(f)

    if experiments_file.exists():
        import pandas as pd
        df = pd.read_csv(experiments_file)
        response["experiments"] = df.to_dict(orient="records")

    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No saved evaluation metrics found. Run 'python -m src.deeprec.evaluation.experiments' first.",
        )

    return response
