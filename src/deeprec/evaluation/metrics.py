"""Recommendation evaluation metrics: Recall@K, Precision@K, NDCG@K, MAP@K, Coverage."""

from typing import List, Set, Union
import numpy as np


def precision_at_k(recommended: List[int], relevant: Set[int] | List[int], k: int) -> float:
    """Calculate Precision@K.
    
    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set or list of ground truth relevant item IDs.
        k: Cut-off rank (must be > 0).
        
    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not recommended or not relevant:
        return 0.0

    # Handle duplicates by taking first k unique
    seen = set()
    deduped = []
    for item in recommended:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
        if len(deduped) == k:
            break

    top_k = deduped[:k]
    rel_set = set(relevant)
    hits = sum(1 for item in top_k if item in rel_set)
    return float(hits / k)


def recall_at_k(recommended: List[int], relevant: Set[int] | List[int], k: int) -> float:
    """Calculate Recall@K.
    
    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set or list of ground truth relevant item IDs.
        k: Cut-off rank.
        
    Returns:
        Recall score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not recommended or not relevant:
        return 0.0

    rel_set = set(relevant)
    if len(rel_set) == 0:
        return 0.0

    seen = set()
    deduped = []
    for item in recommended:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
        if len(deduped) == k:
            break

    top_k = deduped[:k]
    hits = sum(1 for item in top_k if item in rel_set)
    return float(hits / len(rel_set))


def ndcg_at_k(recommended: List[int], relevant: Set[int] | List[int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    Uses standard formula: DCG = sum_{r=1}^k (rel_r / log2(r + 1))
    
    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Ground truth relevant item IDs.
        k: Cut-off rank.
        
    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not recommended or not relevant:
        return 0.0

    rel_set = set(relevant)
    if len(rel_set) == 0:
        return 0.0

    seen = set()
    deduped = []
    for item in recommended:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
        if len(deduped) == k:
            break

    # Calculate DCG
    dcg = 0.0
    for r, item in enumerate(deduped[:k]):
        if item in rel_set:
            dcg += 1.0 / np.log2(r + 2.0) # r is 0-indexed, so r+2 gives rank r+1

    # Calculate IDCG (ideal DCG)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(r + 2.0) for r in range(ideal_hits))

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def map_at_k(recommended: List[int], relevant: Set[int] | List[int], k: int) -> float:
    """Calculate Mean Average Precision at K (MAP@K) for a single user.
    
    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Ground truth relevant item IDs.
        k: Cut-off rank.
        
    Returns:
        Average precision score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not recommended or not relevant:
        return 0.0

    rel_set = set(relevant)
    if len(rel_set) == 0:
        return 0.0

    seen = set()
    deduped = []
    for item in recommended:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
        if len(deduped) == k:
            break

    running_hits = 0
    score_sum = 0.0
    for r, item in enumerate(deduped[:k]):
        if item in rel_set:
            running_hits += 1
            score_sum += running_hits / (r + 1.0)

    denominator = min(len(rel_set), k)
    if denominator == 0:
        return 0.0
    return float(score_sum / denominator)


def catalog_coverage(all_recommended_lists: List[List[int]], total_catalog_size: int) -> float:
    """Calculate catalog coverage as fraction of total catalogue recommended.
    
    Args:
        all_recommended_lists: List of recommendation lists across users.
        total_catalog_size: Total number of unique items in the catalog.
        
    Returns:
        Coverage ratio between 0.0 and 1.0.
    """
    if total_catalog_size <= 0:
        return 0.0

    unique_recommended = set()
    for rec_list in all_recommended_lists:
        for item in rec_list:
            if item != 0:
                unique_recommended.add(item)

    return float(len(unique_recommended) / total_catalog_size)
