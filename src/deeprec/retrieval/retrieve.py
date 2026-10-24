"""Candidate retrieval engine supporting batched scoring, seen-item filtering, and fallback."""

from typing import List, Tuple, Dict, Any, Optional
import torch
import numpy as np


class CandidateRetriever:
    """Retrieval engine for Candidate Generation stage."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_items: int,
        device: torch.device,
        popular_items: Optional[List[int]] = None,
    ):
        self.model = model
        self.num_items = num_items
        self.device = device
        self.popular_items = popular_items or list(range(1, min(101, num_items + 1)))
        self.model.eval()

    @torch.no_grad()
    def retrieve(
        self,
        history: List[int] | np.ndarray | torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
        k: int = 100,
        exclude_seen: bool = True,
        seen_items: Optional[set | list] = None,
    ) -> Tuple[List[int], List[float]]:
        """Retrieve top-K candidate items for a single user given history.
        
        Args:
            history: Sequence of previous movie IDs (can include 0 padding).
            context: Optional demographic features dict.
            k: Number of candidates to retrieve.
            exclude_seen: If True, exclude items in history/seen_items.
            seen_items: Set of item IDs already seen by user.
            
        Returns:
            candidate_ids: List of retrieved item IDs (length <= k).
            candidate_scores: List of model logits for retrieved candidates.
        """
        # Ensure tensor
        if isinstance(history, list) or isinstance(history, np.ndarray):
            hist_tensor = torch.tensor(history, dtype=torch.long, device=self.device)
        else:
            hist_tensor = history.to(self.device)

        if hist_tensor.dim() == 1:
            hist_tensor = hist_tensor.unsqueeze(0) # [1, L]

        # Check for cold start (all zeros or empty)
        non_zero_items = hist_tensor[hist_tensor != 0]
        if len(non_zero_items) == 0:
            # Fallback to popular items
            return self._fallback_popular(k=k, seen_items=seen_items)

        # Encode user vector
        user_vector = self.model.encode(hist_tensor, context=context) # [1, D]

        # Vectorized scoring over full catalogue: [1, num_items + 1]
        all_logits = self.model.score_all(user_vector).squeeze(0) # [num_items + 1]

        # Filter seen items
        if exclude_seen:
            seen_set = set(non_zero_items.cpu().numpy().tolist())
            if seen_items:
                seen_set.update(seen_items)
            for s_id in seen_set:
                if 0 <= s_id < len(all_logits):
                    all_logits[s_id] = -1e9

        # Ensure index 0 (padding) is never retrieved
        all_logits[0] = -1e9

        # Top-K
        retrieval_k = min(k, self.num_items)
        scores, indices = torch.topk(all_logits, k=retrieval_k)

        cand_ids = indices.cpu().tolist()
        cand_scores = scores.cpu().tolist()

        return cand_ids, cand_scores

    def _fallback_popular(
        self,
        k: int,
        seen_items: Optional[set | list] = None,
    ) -> Tuple[List[int], List[float]]:
        """Fallback recommendation of most popular items for cold-start users."""
        seen = set(seen_items) if seen_items else set()
        cands = [m for m in self.popular_items if m not in seen][:k]
        # Assign decaying synthetic score for baseline ranking
        scores = [1.0 / (i + 1) for i in range(len(cands))]
        return cands, scores

    @torch.no_grad()
    def retrieve_batch(
        self,
        histories: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
        k: int = 100,
        exclude_seen: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched retrieval for multiple users simultaneously (used in fast evaluation).
        
        Args:
            histories: [B, max_len]
            context: Optional context dict
            k: Top-K candidates
            
        Returns:
            candidate_ids: LongTensor of shape [B, k]
            candidate_scores: FloatTensor of shape [B, k]
        """
        histories = histories.to(self.device)
        user_vectors = self.model.encode(histories, context=context) # [B, D]
        all_logits = self.model.score_all(user_vectors)              # [B, num_items + 1]

        if exclude_seen:
            # Mask all items in history
            mask_idx = histories.clone()
            # Don't mask 0 with -1e9 because 0 is already -1e9
            all_logits.scatter_(1, mask_idx, -1e9)

        all_logits[:, 0] = -1e9

        retrieval_k = min(k, self.num_items)
        scores, indices = torch.topk(all_logits, k=retrieval_k, dim=1)
        return indices, scores
