"""End-to-end two-stage recommendation service orchestrator."""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle
import numpy as np
import pandas as pd
import torch

from src.deeprec.config import load_config, get_device
from src.deeprec.models.candidate_generation import CandidateGenerator
from src.deeprec.models.ranking import DeepRankingModel
from src.deeprec.data.ranking_features import RankingFeatureExtractor
from src.deeprec.retrieval.retrieve import CandidateRetriever


class RecommendationService:
    """Orchestrates candidate generation, feature extraction, and ranking."""

    def __init__(self, config_path: str = "configs/model.yaml"):
        self.config = load_config(config_path)
        self.device = get_device(self.config.get("system", {}).get("device", "auto"))
        self.paths = self.config["paths"]

        self.candidate_model: Optional[CandidateGenerator] = None
        self.ranking_model: Optional[DeepRankingModel] = None
        self.feature_extractor: Optional[RankingFeatureExtractor] = None
        self.retriever: Optional[CandidateRetriever] = None

        self.movie_catalogue: Dict[int, Dict[str, Any]] = {}
        self.movie_map: Dict[int, int] = {} # raw movie_id -> movie_idx
        self.user_map: Dict[int, int] = {}  # raw user_id -> user_idx
        self.idx_to_movie: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}

        self.user_histories: Dict[int, List[int]] = {}
        self.user_seen_items: Dict[int, set] = {}
        self.popular_items: List[int] = []

        # In-memory buffer for newly recorded interactions
        self.recent_interactions: List[Dict[str, Any]] = []

        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """Load metadata, datasets, and trained model weights into memory."""
        try:
            data_dir = Path(self.paths["processed_data_dir"])
            cand_path = Path(self.paths["candidate_model_path"])
            rank_path = Path(self.paths["ranking_model_path"])
            rank_artifact_dir = Path(self.paths["ranking_artifact_dir"])

            if not (data_dir / "movies.csv").exists():
                print(f"[DeepRec] Processed data not found in {data_dir}.")
                return False

            # Load mappings
            with open(data_dir / "movie_map.json") as f:
                raw_movie_map = json.load(f)
                self.movie_map = {int(k): int(v) for k, v in raw_movie_map.items()}
                self.idx_to_movie = {v: k for k, v in self.movie_map.items()}

            with open(data_dir / "user_map.json") as f:
                raw_user_map = json.load(f)
                self.user_map = {int(k): int(v) for k, v in raw_user_map.items()}
                self.idx_to_user = {v: k for k, v in self.user_map.items()}

            # Load movies metadata catalogue
            movies_df = pd.read_csv(data_dir / "movies.csv")
            genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
            for _, row in movies_df.iterrows():
                m_idx = int(row["movie_idx"])
                genres = [c.replace("genre_", "") for c in genre_cols if row.get(c, 0) == 1]
                self.movie_catalogue[m_idx] = {
                    "movie_idx": m_idx,
                    "movie_id": int(row["movie_id"]),
                    "title": str(row["title"]),
                    "genres": genres,
                    "release_year": int(row.get("release_year", 1990)),
                }

            # Build user interaction histories from train and validation
            train_df = pd.read_csv(data_dir / "train.csv")
            val_df = pd.read_csv(data_dir / "validation.csv")
            combined_df = pd.concat([train_df, val_df]).sort_values(["user_idx", "timestamp"])

            for u_idx, group in combined_df.groupby("user_idx"):
                hist = group["movie_idx"].tolist()
                self.user_histories[int(u_idx)] = hist
                self.user_seen_items[int(u_idx)] = set(hist)

            self.popular_items = train_df["movie_idx"].value_counts().index.tolist()

            # Load Candidate Generator
            if cand_path.exists():
                cand_chk = torch.load(cand_path, map_location=self.device, weights_only=False)
                self.candidate_model = CandidateGenerator(
                    num_items=cand_chk["num_items"],
                    embedding_dim=cand_chk["embedding_dim"],
                    hidden_dims=cand_chk["hidden_dims"],
                    use_context=cand_chk.get("use_context", False),
                ).to(self.device)
                self.candidate_model.load_state_dict(cand_chk["model_state_dict"], strict=False)
                self.candidate_model.eval()

                self.retriever = CandidateRetriever(
                    model=self.candidate_model,
                    num_items=cand_chk["num_items"],
                    device=self.device,
                    popular_items=self.popular_items,
                )
            else:
                print(f"[DeepRec] WARNING: Candidate model checkpoint not found at {cand_path}.")

            # Load Ranking Model and Feature Extractor
            extractor_file = rank_artifact_dir / "feature_extractor.pkl"
            if extractor_file.exists():
                with open(extractor_file, "rb") as f:
                    self.feature_extractor = pickle.load(f)

            if rank_path.exists():
                rank_chk = torch.load(rank_path, map_location=self.device, weights_only=False)
                self.ranking_model = DeepRankingModel(
                    num_users=rank_chk["num_users"],
                    num_movies=rank_chk["num_movies"],
                    continuous_dim=rank_chk["continuous_dim"],
                    embedding_dim=rank_chk["embedding_dim"],
                    hidden_dims=rank_chk["hidden_dims"],
                ).to(self.device)
                self.ranking_model.load_state_dict(rank_chk["model_state_dict"])
                self.ranking_model.eval()
            else:
                print(f"[DeepRec] WARNING: Ranking model checkpoint not found at {rank_path}.")

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[DeepRec] Error loading artifacts: {e}")
            return False

    def get_recommendations(
        self,
        user_id: int,
        candidate_k: int = 100,
        top_n: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[str, Any]:
        """Generate two-stage recommendations for a given raw user_id."""
        if not self.is_loaded:
            raise RuntimeError("Models and metadata are not loaded. Run training pipelines first.")

        # Resolve user_idx
        is_cold_start = user_id not in self.user_map
        user_idx = self.user_map.get(user_id, 0)
        user_history = self.user_histories.get(user_idx, [])
        seen_items = self.user_seen_items.get(user_idx, set())

        # If user is completely unknown or has no history, provide cold-start fallback
        if is_cold_start or len(user_history) == 0:
            return self._cold_start_recommendations(user_id, top_n)

        # STAGE 1: Candidate Generation
        max_history_len = self.config.get("data", {}).get("max_history_length", 20)
        recent_history = user_history[-max_history_len:]
        padded_history = [0] * (max_history_len - len(recent_history)) + recent_history

        if self.retriever is not None:
            cands, cand_scores = self.retriever.retrieve(
                history=padded_history,
                k=candidate_k,
                exclude_seen=exclude_seen,
                seen_items=seen_items if exclude_seen else None,
            )
        else:
            # Fallback to popular items
            cands = [m for m in self.popular_items if m not in seen_items][:candidate_k]
            cand_scores = [1.0 / (i + 1) for i in range(len(cands))]

        # STAGE 2: Neural Ranking
        if self.ranking_model is not None and self.feature_extractor is not None:
            raw_feats = []
            for c_id, c_s in zip(cands, cand_scores):
                feat = self.feature_extractor.extract_raw_features(
                    user_idx=user_idx, movie_idx=c_id, candidate_score=c_s
                )
                raw_feats.append(feat)

            trans_feats = self.feature_extractor.transform_features(np.array(raw_feats, dtype=np.float32))
            ranked_pairs = self.ranking_model.rank_candidates(
                user_id=user_idx,
                candidate_ids=cands,
                continuous_features=trans_feats,
                device=self.device,
            )
            # Take top_n
            final_recs = ranked_pairs[:top_n]
            # Map candidate scores
            cand_score_dict = dict(zip(cands, cand_scores))
        else:
            # If ranking model not present, sort by Stage 1 candidate scores
            pairs = sorted(zip(cands, cand_scores), key=lambda x: x[1], reverse=True)
            final_recs = pairs[:top_n]
            cand_score_dict = dict(pairs)

        # Construct enriched response
        recommendations = []
        for rank, (m_idx, r_score) in enumerate(final_recs, 1):
            info = self.movie_catalogue.get(m_idx, {})
            c_score = cand_score_dict.get(m_idx, 0.0)

            # Generate evidence-based explanation
            explanations = []
            if self.feature_extractor is not None:
                u_genre = self.feature_extractor.user_genre_profiles.get(user_idx)
                m_genre = self.feature_extractor.movie_genre_vectors.get(m_idx)
                if u_genre is not None and m_genre is not None and np.dot(u_genre, m_genre) > 0.4:
                    explanations.append(f"Strong genre affinity with your watch history")
            if m_idx in self.popular_items[:50]:
                explanations.append("High overall platform engagement")
            if c_score > 0.5:
                explanations.append("Strong sequential relevance in Stage 1")
            if not explanations:
                explanations.append("Ranked by deep user-item engagement model")

            recommendations.append({
                "rank": rank,
                "movie_id": info.get("movie_id", self.idx_to_movie.get(m_idx, m_idx)),
                "movie_idx": m_idx,
                "title": info.get("title", f"Movie #{m_idx}"),
                "genres": info.get("genres", []),
                "release_year": info.get("release_year", 1990),
                "candidate_score": round(float(c_score), 4),
                "ranking_score": round(float(r_score), 4),
                "explanation": "; ".join(explanations),
            })

        # User recent history for display
        history_display = []
        for h_idx in user_history[-5:]:
            h_info = self.movie_catalogue.get(h_idx, {})
            history_display.append({
                "movie_id": h_info.get("movie_id", h_idx),
                "title": h_info.get("title", f"Movie #{h_idx}"),
                "genres": h_info.get("genres", []),
            })

        return {
            "user_id": user_id,
            "user_idx": user_idx,
            "is_cold_start": False,
            "candidate_count": len(cands),
            "returned_count": len(recommendations),
            "history": history_display,
            "recommendations": recommendations,
        }

    def _cold_start_recommendations(self, user_id: int, top_n: int) -> Dict[str, Any]:
        """Popularity-based fallback recommendation for cold-start users."""
        recs = []
        for rank, m_idx in enumerate(self.popular_items[:top_n], 1):
            info = self.movie_catalogue.get(m_idx, {})
            recs.append({
                "rank": rank,
                "movie_id": info.get("movie_id", self.idx_to_movie.get(m_idx, m_idx)),
                "movie_idx": m_idx,
                "title": info.get("title", f"Movie #{m_idx}"),
                "genres": info.get("genres", []),
                "release_year": info.get("release_year", 1990),
                "candidate_score": round(1.0 / rank, 4),
                "ranking_score": round(1.0 / rank, 4),
                "explanation": "Cold-start fallback: Top trending catalogue movie",
            })

        return {
            "user_id": user_id,
            "user_idx": 0,
            "is_cold_start": True,
            "candidate_count": len(recs),
            "returned_count": len(recs),
            "history": [],
            "recommendations": recs,
        }

    def record_interaction(
        self,
        user_id: int,
        movie_id: int,
        interaction_type: str = "watch",
        rating: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record a new interaction dynamically in the in-memory buffer."""
        m_idx = self.movie_map.get(movie_id)
        if m_idx is None:
            # Check if movie_id was already passed as movie_idx
            if movie_id in self.movie_catalogue:
                m_idx = movie_id
            else:
                raise ValueError(f"Movie ID {movie_id} does not exist in catalogue.")

        user_idx = self.user_map.get(user_id)
        if user_idx is None:
            # Assign next user_idx dynamically
            user_idx = len(self.user_map) + 1
            self.user_map[user_id] = user_idx
            self.idx_to_user[user_idx] = user_id
            self.user_histories[user_idx] = []
            self.user_seen_items[user_idx] = set()

        # Update in-memory state
        self.user_histories[user_idx].append(m_idx)
        self.user_seen_items[user_idx].add(m_idx)

        record = {
            "user_id": user_id,
            "user_idx": user_idx,
            "movie_id": movie_id,
            "movie_idx": m_idx,
            "interaction_type": interaction_type,
            "rating": rating,
            "timestamp": time.time(),
        }
        self.recent_interactions.append(record)

        return {
            "status": "recorded",
            "message": f"Recorded {interaction_type} for user {user_id} and movie {movie_id}",
            "interaction": record,
            "total_user_interactions": len(self.user_histories[user_idx]),
        }
