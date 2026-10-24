from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepRankingModel(nn.Module):
    """Deep Ranking Network for scoring and reranking candidates.
    
    Inspired by 'Deep Neural Networks for YouTube Recommendations' (Covington et al., 2016):
    - Embeds categorical IDs (user_idx, movie_idx).
    - Takes continuous candidate-level features (popularity, recency, ratings, Stage 1 candidate score,
      genre overlap, and power transforms x, x^2, sqrt(x)).
    - Dense feed-forward layers with ReLU activations.
    - Produces a single engagement logit for ordering the final recommendation feed.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        continuous_dim: int = 27,
        embedding_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.continuous_dim = continuous_dim
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Categorical Embeddings
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)

        # Total input dimension to MLP
        input_dim = (2 * embedding_dim) + continuous_dim

        # Dense Tower
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final scoring head: produces a single scalar logit
        self.mlp = nn.Sequential(*layers)
        self.score_head = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0)
            self.movie_embedding.weight[0].fill_(0)
        nn.init.xavier_uniform_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        continuous_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to score (user, movie) candidate pairs.
        
        Args:
            user_ids: LongTensor of shape [batch_size]
            movie_ids: LongTensor of shape [batch_size]
            continuous_features: FloatTensor of shape [batch_size, continuous_dim]
            
        Returns:
            scores: FloatTensor of shape [batch_size]
        """
        # Clamp IDs safely to valid bounds
        safe_user_ids = torch.clamp(user_ids, 0, self.num_users)
        safe_movie_ids = torch.clamp(movie_ids, 0, self.num_movies)

        u_emb = self.user_embedding(safe_user_ids)   # [B, D]
        m_emb = self.movie_embedding(safe_movie_ids) # [B, D]

        # Combine embeddings and continuous features
        combined = torch.cat([u_emb, m_emb, continuous_features], dim=-1)

        # Pass through dense layers
        latent = self.mlp(combined)
        logit = self.score_head(latent).squeeze(-1) # [B]
        return logit

    @torch.no_grad()
    def rank_candidates(
        self,
        user_id: int,
        candidate_ids: List[int],
        continuous_features: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tuple[int, float]]:
        """Score and sort candidate items for a given user.
        
        Args:
            user_id: User index.
            candidate_ids: List of candidate movie indices.
            continuous_features: 2D numpy array [len(candidate_ids), continuous_dim].
            device: Compute device.
            
        Returns:
            Sorted list of (movie_id, ranking_score) pairs in descending order.
        """
        if not candidate_ids:
            return []

        self.eval()
        n = len(candidate_ids)
        u_tensor = torch.full((n,), user_id, dtype=torch.long, device=device)
        m_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        f_tensor = torch.tensor(continuous_features, dtype=torch.float32, device=device)

        logits = self.forward(u_tensor, m_tensor, f_tensor)
        scores = torch.sigmoid(logits).cpu().tolist()

        pairs = list(zip(candidate_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
