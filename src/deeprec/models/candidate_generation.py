"""Deep Candidate Generation Neural Network based on Covington et al. (YouTube 2016)."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateGenerator(nn.Module):
    """Two-Tower Sequence-to-Item Deep Candidate Generation Model.
    
    Inspired by 'Deep Neural Networks for YouTube Recommendations' (Covington et al., 2016):
    - Sparse video watch history -> Learned dense embeddings -> Masked Average Pooling.
    - Context features (User demographic / continuous features) concatenated with history.
    - Multiple dense layers with ReLU activations to produce user representation vector u.
    - Item candidates represented by output embeddings V and output biases b.
    - Scoring via dot product: s(u, v) = u^T v + b.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        use_context: bool = True,
        num_genders: int = 2,
        num_occupations: int = 22,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_context = use_context

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # 1. Item History Embedding: Index 0 is reserved for PADDING
        self.history_embedding = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=0
        )

        # 2. Output Item Representations: Separate learned embeddings for candidate scoring
        self.output_embedding = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=0
        )
        self.output_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)

        # 3. Context Feature Embeddings
        total_input_dim = embedding_dim
        if self.use_context:
            self.gender_embedding = nn.Embedding(num_genders + 1, 8)
            self.occ_embedding = nn.Embedding(num_occupations + 1, 16)
            self.age_projection = nn.Linear(1, 8)
            total_input_dim += 8 + 16 + 8 # embedding_dim + 32

        # 4. Dense Tower Layers
        layers = []
        prev_dim = total_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection to match embedding_dim for dot-product scoring
        if prev_dim != embedding_dim:
            layers.append(nn.Linear(prev_dim, embedding_dim))
            layers.append(nn.LayerNorm(embedding_dim))

        self.user_tower = nn.Sequential(*layers)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.history_embedding.weight, std=0.01)
        nn.init.normal_(self.output_embedding.weight, std=0.01)
        nn.init.zeros_(self.output_bias.weight)
        # Ensure padding embedding stays zero
        with torch.no_grad():
            self.history_embedding.weight[0].fill_(0)
            self.output_embedding.weight[0].fill_(0)

    def encode(
        self,
        history: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode user watch history and context into a user embedding vector u.
        
        Args:
            history: Tensor of shape [batch_size, max_history_len] with item IDs.
            context: Optional dictionary with 'gender', 'occupation', 'age' tensors.
            
        Returns:
            user_vector: Tensor of shape [batch_size, embedding_dim].
        """
        # 1. Embed history: [B, L, D]
        item_embeds = self.history_embedding(history)

        # 2. Masked Average Pooling: exclude padding_idx (0)
        # mask shape: [B, L, 1]
        mask = (history != 0).unsqueeze(-1).float()
        summed = (item_embeds * mask).sum(dim=1) # [B, D]
        counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        user_repr = summed / counts              # [B, D]

        # 3. Concatenate context features if enabled
        if self.use_context:
            device = history.device
            batch_size = history.shape[0]

            if context is not None and "gender" in context:
                gender_emb = self.gender_embedding(context["gender"])
            else:
                gender_emb = torch.zeros(batch_size, 8, device=device)

            if context is not None and "occupation" in context:
                occ_emb = self.occ_embedding(context["occupation"])
            else:
                occ_emb = torch.zeros(batch_size, 16, device=device)

            if context is not None and "age" in context:
                age_in = context["age"].unsqueeze(-1) if context["age"].dim() == 1 else context["age"]
                age_feat = F.relu(self.age_projection(age_in))
            else:
                age_feat = torch.zeros(batch_size, 8, device=device)

            user_repr = torch.cat([user_repr, gender_emb, occ_emb, age_feat], dim=-1)

        # 4. Deep MLP Tower
        return self.user_tower(user_repr)

    def score(self, user_vectors: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Score specific candidate items for each user.
        
        Args:
            user_vectors: [B, D]
            candidate_ids: [B, C] or [B]
            
        Returns:
            scores: [B, C] or [B]
        """
        if candidate_ids.dim() == 1:
            candidate_ids = candidate_ids.unsqueeze(1)
            single_item = True
        else:
            single_item = False

        # [B, C, D]
        item_vectors = self.output_embedding(candidate_ids)
        # [B, C]
        bias = self.output_bias(candidate_ids).squeeze(-1)

        # Dot product: user_vectors.unsqueeze(1) is [B, 1, D]
        logits = (user_vectors.unsqueeze(1) * item_vectors).sum(dim=-1) + bias

        if single_item:
            return logits.squeeze(1)
        return logits

    def score_all(self, user_vectors: torch.Tensor) -> torch.Tensor:
        """Vectorized full-catalogue scoring for candidate retrieval.
        
        Args:
            user_vectors: [B, D]
            
        Returns:
            all_scores: [B, num_items + 1]
        """
        # output_embedding.weight: [num_items + 1, D]
        # output_bias.weight: [num_items + 1, 1]
        logits = torch.matmul(user_vectors, self.output_embedding.weight.t()) + self.output_bias.weight.t()
        # Disallow padding index 0 from ever being selected
        logits[:, 0] = -1e9
        return logits
