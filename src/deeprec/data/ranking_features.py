"""Feature engineering and normalization for the Deep Ranking Model."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class RankingFeatureExtractor:
    """Extracts candidate-level continuous and categorical features.
    
    Adheres strictly to research paper principles:
    - Statistics fitted on training interactions ONLY.
    - Continuous features normalized and augmented with powers: x, x^2, sqrt(x).
    - User-item interaction recency, candidate generation scores, and genre affinity.
    """

    def __init__(self, genres: Optional[List[str]] = None):
        self.genres = genres or [
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        self.num_genres = len(self.genres)

        # Statistics fitted on train data
        self.item_popularity: Dict[int, float] = {}
        self.item_avg_rating: Dict[int, float] = {}
        self.item_rating_count: Dict[int, float] = {}
        self.global_mean_rating: float = 3.5

        self.user_mean_rating: Dict[int, float] = {}
        self.user_interaction_count: Dict[int, float] = {}
        self.user_genre_profiles: Dict[int, np.ndarray] = {}
        self.user_last_timestamp: Dict[int, float] = {}

        self.movie_genre_vectors: Dict[int, np.ndarray] = {}
        self.movie_release_years: Dict[int, float] = {}

        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> "RankingFeatureExtractor":
        """Compute all baseline statistics and feature normalization strictly on train data."""
        # 1. Movie genre vectors
        genre_cols = [f"genre_{g}" for g in self.genres if f"genre_{g}" in movies_df.columns]
        if not genre_cols:
            genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]

        for _, row in movies_df.iterrows():
            midx = int(row["movie_idx"])
            g_vec = row[genre_cols].values.astype(np.float32)
            norm = np.linalg.norm(g_vec)
            if norm > 0:
                g_vec = g_vec / norm
            self.movie_genre_vectors[midx] = g_vec
            self.movie_release_years[midx] = float(row.get("release_year", 1990))

        # 2. Item statistics from train_df
        self.global_mean_rating = float(train_df["rating"].mean())
        item_stats = train_df.groupby("movie_idx")["rating"].agg(["count", "mean"]).reset_index()
        max_count = float(train_df["movie_idx"].value_counts().max())

        for _, row in item_stats.iterrows():
            midx = int(row["movie_idx"])
            cnt = float(row["count"])
            mean_r = float(row["mean"])
            self.item_popularity[midx] = cnt / max(1.0, max_count)
            self.item_rating_count[midx] = cnt
            # Bayesian smoothed rating
            self.item_avg_rating[midx] = (cnt * mean_r + 5.0 * self.global_mean_rating) / (cnt + 5.0)

        # 3. User statistics from train_df
        user_stats = train_df.groupby("user_idx")["rating"].agg(["count", "mean"]).reset_index()
        for _, row in user_stats.iterrows():
            uidx = int(row["user_idx"])
            self.user_mean_rating[uidx] = float(row["mean"])
            self.user_interaction_count[uidx] = float(row["count"])

        # 4. User genre profiles and last interaction timestamp
        train_sorted = train_df.sort_values(["user_idx", "timestamp"])
        for uidx, group in train_sorted.groupby("user_idx"):
            self.user_last_timestamp[int(uidx)] = float(group["timestamp"].max())
            # Aggregate watched movie genres
            watched_genres = np.zeros(len(genre_cols), dtype=np.float32)
            for midx in group["movie_idx"]:
                if int(midx) in self.movie_genre_vectors:
                    watched_genres += self.movie_genre_vectors[int(midx)]
            norm = np.linalg.norm(watched_genres)
            if norm > 0:
                watched_genres /= norm
            self.user_genre_profiles[int(uidx)] = watched_genres

        self.is_fitted = True
        return self

    def extract_raw_features(
        self,
        user_idx: int,
        movie_idx: int,
        candidate_score: float = 0.0,
        current_timestamp: Optional[float] = None,
    ) -> np.ndarray:
        """Extract continuous feature vector for a single user-candidate pair."""
        # 1. Candidate score from Stage 1
        cand_s = float(candidate_score)

        # 2. Item features
        pop = self.item_popularity.get(movie_idx, 0.0)
        item_rating = self.item_avg_rating.get(movie_idx, self.global_mean_rating)
        item_cnt = np.log1p(self.item_rating_count.get(movie_idx, 0.0))

        # 3. User features
        u_rating = self.user_mean_rating.get(user_idx, self.global_mean_rating)
        u_cnt = np.log1p(self.user_interaction_count.get(user_idx, 1.0))

        # 4. User-item genre affinity (cosine similarity)
        u_genre = self.user_genre_profiles.get(user_idx, None)
        m_genre = self.movie_genre_vectors.get(movie_idx, None)
        if u_genre is not None and m_genre is not None:
            genre_sim = float(np.dot(u_genre, m_genre))
        else:
            genre_sim = 0.0

        # 5. Recency / Example age feature
        last_ts = self.user_last_timestamp.get(user_idx, current_timestamp or 0.0)
        if current_timestamp is not None and current_timestamp >= last_ts:
            delta_days = (current_timestamp - last_ts) / 86400.0
            recency = np.log1p(max(0.0, delta_days))
        else:
            recency = 0.0

        # 6. Movie release year relative to 1998 (MovieLens baseline year)
        rel_year = (self.movie_release_years.get(movie_idx, 1990) - 1920) / 80.0

        # Raw continuous features (8 values)
        return np.array([
            cand_s, pop, item_rating, item_cnt,
            u_rating, u_cnt, genre_sim, recency, rel_year
        ], dtype=np.float32)

    def fit_scaler(self, raw_features_matrix: np.ndarray) -> None:
        """Fit StandardScaler on training feature matrix."""
        self.scaler = StandardScaler()
        self.scaler.fit(raw_features_matrix)

    def transform_features(self, raw_features_matrix: np.ndarray) -> np.ndarray:
        """Apply normalization and augment with non-linear power features (x, x^2, sqrt(x)).
        
        As described in Covington et al. (Section 4.1):
        'We also feed powers of the continuous features x^2 and sqrt(x), giving the network
        expressive power to easily form non-linear combinations.'
        """
        if self.scaler is not None:
            normalized = self.scaler.transform(raw_features_matrix)
        else:
            normalized = raw_features_matrix

        # Shift to positive range for sqrt
        # Map roughly around 0 into positive offset
        pos_x = np.clip(normalized - normalized.min(axis=0, keepdims=True) + 1e-4, 1e-5, None)
        x_squared = normalized ** 2
        x_sqrt = np.sqrt(pos_x)

        # Concatenate [x, x^2, sqrt(x)]
        return np.concatenate([normalized, x_squared, x_sqrt], axis=1).astype(np.float32)

    def get_feature_names(self) -> List[str]:
        base_names = [
            "candidate_score", "item_popularity", "item_avg_rating", "item_rating_count_log",
            "user_mean_rating", "user_count_log", "genre_affinity", "interaction_recency", "release_year"
        ]
        return base_names + [f"{n}_pow2" for n in base_names] + [f"{n}_sqrt" for n in base_names]
