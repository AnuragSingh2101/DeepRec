"""Loss functions for candidate generation and ranking training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampledSoftmaxLoss(nn.Module):
    """Sampled Softmax Loss for extreme multiclass classification.
    
    As described in Covington et al. (Section 3.1):
    Instead of calculating the partition function over the entire catalog,
    we compute the softmax cross-entropy over the positive item plus K sampled negatives.
    
    The positive item is placed at index 0, so the ground-truth target label is 0.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate sampled softmax cross-entropy loss.
        
        Args:
            pos_scores: Tensor of shape [batch_size] or [batch_size, 1]
            neg_scores: Tensor of shape [batch_size, num_negatives]
            
        Returns:
            Scalar loss tensor.
        """
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(1)

        # Concatenate: positive is at column 0, negatives at columns 1..K
        # Shape: [batch_size, 1 + num_negatives]
        all_logits = torch.cat([pos_scores, neg_scores], dim=1)

        # Ground truth is index 0 for every sample
        labels = torch.zeros(all_logits.shape[0], dtype=torch.long, device=all_logits.device)
        return self.ce(all_logits, labels)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Pairwise Loss: -log(sigmoid(s_pos - s_neg))."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            pos_scores: [B] or [B, 1]
            neg_scores: [B, num_negatives]
        """
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(1)
        # Difference: [B, K]
        diff = pos_scores - neg_scores
        loss = -torch.mean(F.logsigmoid(diff))
        return loss
