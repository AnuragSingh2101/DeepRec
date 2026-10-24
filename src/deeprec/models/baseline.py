"""Baseline recommendation models: Popularity and Item-Item Collaborative Filtering."""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict


class PopularityRecommender:
    """Popularity-based recommendation baseline.
    
    Ranks items strictly by their interaction frequency in the training set.
    """

    def __init__(self):
        self.popular_items: List[int] = []
        self.popularity_scores: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """Compute item frequencies from training interactions."""
        counts = train_df["movie_idx"].value_counts()
        max_count = float(counts.max()) if len(counts) > 0 else 1.0

        self.popular_items = counts.index.tolist()
        self.popularity_scores = {int(m): float(c / max_count) for m, c in counts.items()}
        return self

    def recommend(
        self,
        user_history: List[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """Recommend top-k most popular unseen items."""
        seen = set(user_history) if exclude_seen else set()
        recommendations = []
        scores = []

        for item in self.popular_items:
            if item not in seen and item != 0:
                recommendations.append(item)
                scores.append(self.popularity_scores.get(item, 0.0))
                if len(recommendations) >= k:
                    break

        return recommendations, scores


class ItemItemCFRecommender:
    """Lightweight Item-Item Collaborative Filtering baseline.
    
    Computes cosine similarity between item co-occurrence vectors in training set.
    """

    def __init__(self, top_similar_k: int = 20):
        self.top_similar_k = top_similar_k
        self.similarity_matrix: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.popular_fallback: List[int] = []

    def fit(self, train_df: pd.DataFrame) -> "ItemItemCFRecommender":
        """Build item co-occurrence and cosine similarities from training interactions."""
        # Precompute popularity fallback
        counts = train_df["movie_idx"].value_counts()
        self.popular_fallback = counts.index.tolist()

        # Build user -> items mapping
        user_items = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

        # Build item co-occurrence
        item_users = train_df.groupby("movie_idx")["user_idx"].apply(set).to_dict()
        item_norms = {m: np.sqrt(len(u_set)) for m, u_set in item_users.items()}

        all_items = list(item_users.keys())
        # For MovieLens 100K catalogue (~1682 items), computing sparse co-occurrence is fast
        for i, item_a in enumerate(all_items):
            users_a = item_users[item_a]
            norm_a = item_norms[item_a]
            if norm_a == 0:
                continue

            scores = []
            for item_b in all_items:
                if item_a == item_b:
                    continue
                users_b = item_users[item_b]
                common = len(users_a.intersection(users_b))
                if common > 0:
                    sim = common / (norm_a * item_norms[item_b])
                    scores.append((item_b, float(sim)))

            # Keep top similar items
            scores.sort(key=lambda x: x[1], reverse=True)
            self.similarity_matrix[item_a] = scores[:self.top_similar_k]

        return self

    def recommend(
        self,
        user_history: List[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """Score candidate items by aggregating similarity to recent items in user history."""
        seen = set(user_history) if exclude_seen else set()
        valid_history = [m for m in user_history if m != 0]

        if not valid_history:
            # Fallback to popularity
            recs = [m for m in self.popular_fallback if m not in seen][:k]
            scores = [1.0 / (i + 1) for i in range(len(recs))]
            return recs, scores

        item_scores: Dict[int, float] = defaultdict(float)
        # Use recent items with decay
        for idx, hist_item in enumerate(reversed(valid_history[-10:])):
            decay = 1.0 / (1.0 + 0.1 * idx)
            for sim_item, sim_val in self.similarity_matrix.get(hist_item, []):
                if sim_item not in seen:
                    item_scores[sim_item] += sim_val * decay

        if not item_scores:
            recs = [m for m in self.popular_fallback if m not in seen][:k]
            scores = [1.0 / (i + 1) for i in range(len(recs))]
            return recs, scores

        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        recs = [item for item, _ in sorted_items]
        scores = [score for _, score in sorted_items]
        return recs, scores
