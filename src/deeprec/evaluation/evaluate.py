"""Evaluation module computing Recall@K, Precision@K, NDCG@K, MAP@K, Coverage, and Latency."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch

from src.deeprec.config import load_config, get_device
from src.deeprec.data.sequences import build_eval_sequences
from src.deeprec.models.candidate_generation import CandidateGenerator
from src.deeprec.models.ranking import DeepRankingModel
from src.deeprec.data.ranking_features import RankingFeatureExtractor
from src.deeprec.retrieval.retrieve import CandidateRetriever
from src.deeprec.evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, map_at_k, catalog_coverage
)


def evaluate_system(
    config_path: str = "configs/model.yaml",
    k_values: Optional[List[int]] = None,
    candidate_k: int = 100,
    max_eval_users: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate candidate generation and two-stage recommendation on held-out test data.
    
    Args:
        config_path: Path to configuration file.
        k_values: Cutoff ranks for metrics (e.g. [10, 20, 50]).
        candidate_k: Number of candidates retrieved in Stage 1 before ranking.
        max_eval_users: Optional limit on test users to speed up evaluation.
        
    Returns:
        Dictionary of computed offline metrics.
    """
    cfg = load_config(config_path)
    paths = cfg["paths"]
    device = get_device(cfg.get("system", {}).get("device", "auto"))

    if k_values is None:
        k_values = cfg.get("evaluation", {}).get("k_values", [10, 20, 50])

    data_dir = Path(paths["processed_data_dir"])
    cand_path = Path(paths["candidate_model_path"])
    rank_path = Path(paths["ranking_model_path"])
    eval_dir = Path(paths["evaluation_artifact_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    movies_df = pd.read_csv(data_dir / "movies.csv")
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    num_items = meta["num_movies"]
    num_users = meta["num_users"]
    relevance_th = cfg.get("data", {}).get("relevance_rating_threshold", 4.0)

    # 2. Build test evaluation sequences: history = train + val
    max_len = cfg.get("data", {}).get("max_history_length", 20)
    test_hist, test_tgt, test_u = build_eval_sequences(
        train_df=train_df,
        target_df=test_df,
        max_history_length=max_len,
        padding_idx=0,
        prior_target_df=val_df,
    )

    # Ground truth: map user to relevant test target items (rating >= threshold)
    test_ratings = dict(zip(test_df["user_idx"], test_df["rating"]))
    test_user_relevant = {}
    for i, u in enumerate(test_u):
        r = test_ratings.get(u, 5.0)
        # Held-out relevant item
        if r >= relevance_th:
            test_user_relevant[u] = [int(test_tgt[i])]
        else:
            # Still evaluate retrieval of interacted item
            test_user_relevant[u] = [int(test_tgt[i])]

    # 3. Load Candidate Generator
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate model artifact not found at {cand_path}. Run train_candidates first.")

    cand_chk = torch.load(cand_path, map_location=device, weights_only=False)
    cand_model = CandidateGenerator(
        num_items=cand_chk["num_items"],
        embedding_dim=cand_chk["embedding_dim"],
        hidden_dims=cand_chk["hidden_dims"],
        use_context=cand_chk.get("use_context", False),
    ).to(device)
    cand_model.load_state_dict(cand_chk["model_state_dict"], strict=False)
    cand_model.eval()

    popular_items = train_df["movie_idx"].value_counts().index.tolist()
    retriever = CandidateRetriever(
        model=cand_model,
        num_items=num_items,
        device=device,
        popular_items=popular_items,
    )

    # 4. Load Ranker and Feature Extractor if available
    has_ranker = rank_path.exists()
    rank_model = None
    feature_extractor = None

    if has_ranker:
        import pickle
        extractor_path = Path(paths["ranking_artifact_dir"]) / "feature_extractor.pkl"
        if extractor_path.exists():
            with open(extractor_path, "wb" if not extractor_path.exists() else "rb") as f:
                feature_extractor = pickle.load(f)

        rank_chk = torch.load(rank_path, map_location=device, weights_only=False)
        rank_model = DeepRankingModel(
            num_users=rank_chk["num_users"],
            num_movies=rank_chk["num_movies"],
            continuous_dim=rank_chk["continuous_dim"],
            embedding_dim=rank_chk["embedding_dim"],
            hidden_dims=rank_chk["hidden_dims"],
        ).to(device)
        rank_model.load_state_dict(rank_chk["model_state_dict"])
        rank_model.eval()

    # Pre-index seen items per user
    user_seen = defaultdict_seen = {}
    for u, g in train_df.groupby("user_idx"):
        user_seen[u] = set(g["movie_idx"])
    for u, g in val_df.groupby("user_idx"):
        if u in user_seen:
            user_seen[u].update(g["movie_idx"])
        else:
            user_seen[u] = set(g["movie_idx"])

    # 5. Evaluate Candidate Generation alone vs Two-Stage (Candidate + Ranker)
    eval_users = list(test_user_relevant.keys())
    if max_eval_users:
        eval_users = eval_users[:max_eval_users]

    results = {
        "candidate_only": {f"recall@{k}": [] for k in k_values},
        "two_stage": {f"recall@{k}": [] for k in k_values},
    }
    for k in k_values:
        results["candidate_only"][f"precision@{k}"] = []
        results["candidate_only"][f"ndcg@{k}"] = []
        results["candidate_only"][f"map@{k}"] = []
        results["two_stage"][f"precision@{k}"] = []
        results["two_stage"][f"ndcg@{k}"] = []
        results["two_stage"][f"map@{k}"] = []

    all_cand_recs = []
    all_twostage_recs = []

    retrieval_latencies = []
    ranking_latencies = []

    print("=" * 60)
    print(f"      EVALUATING DEEPREC RECOMMENDATION MODELS")
    print("=" * 60)
    print(f"Evaluation users: {len(eval_users)} | Candidate pool size: {candidate_k}")
    print(f"Ranker enabled: {has_ranker}")
    print("=" * 60)

    # Fast evaluation loop
    for u_idx in eval_users:
        relevant = test_user_relevant[u_idx]
        user_hist = test_hist[test_u == u_idx]
        if len(user_hist) == 0:
            continue
        hist_row = user_hist[0]
        seen = user_seen.get(u_idx, set())

        # Stage 1: Retrieval
        t0 = time.perf_counter()
        cands, cand_scores = retriever.retrieve(
            history=hist_row,
            k=candidate_k,
            exclude_seen=True,
            seen_items=seen,
        )
        t_ret = (time.perf_counter() - t0) * 1000.0
        retrieval_latencies.append(t_ret)

        all_cand_recs.append(cands[:max(k_values)])

        # Compute candidate-only metrics
        for k in k_values:
            results["candidate_only"][f"recall@{k}"].append(recall_at_k(cands, relevant, k))
            results["candidate_only"][f"precision@{k}"].append(precision_at_k(cands, relevant, k))
            results["candidate_only"][f"ndcg@{k}"].append(ndcg_at_k(cands, relevant, k))
            results["candidate_only"][f"map@{k}"].append(map_at_k(cands, relevant, k))

        # Stage 2: Ranking
        if has_ranker and feature_extractor is not None:
            t1 = time.perf_counter()
            # Extract features for candidate items
            raw_feats = []
            for c_id, c_score in zip(cands, cand_scores):
                feat = feature_extractor.extract_raw_features(
                    user_idx=u_idx, movie_idx=c_id, candidate_score=c_score
                )
                raw_feats.append(feat)

            transformed_feats = feature_extractor.transform_features(np.array(raw_feats, dtype=np.float32))
            ranked_pairs = rank_model.rank_candidates(
                user_id=u_idx,
                candidate_ids=cands,
                continuous_features=transformed_feats,
                device=device,
            )
            t_rank = (time.perf_counter() - t1) * 1000.0
            ranking_latencies.append(t_rank)

            reranked_ids = [m for m, s in ranked_pairs]
            all_twostage_recs.append(reranked_ids[:max(k_values)])

            for k in k_values:
                results["two_stage"][f"recall@{k}"].append(recall_at_k(reranked_ids, relevant, k))
                results["two_stage"][f"precision@{k}"].append(precision_at_k(reranked_ids, relevant, k))
                results["two_stage"][f"ndcg@{k}"].append(ndcg_at_k(reranked_ids, relevant, k))
                results["two_stage"][f"map@{k}"].append(map_at_k(reranked_ids, relevant, k))

    # Summarize metrics
    summary = {
        "candidate_generation": {},
        "two_stage_pipeline": {},
        "latency_ms": {
            "retrieval_mean": float(np.mean(retrieval_latencies)) if retrieval_latencies else 0.0,
            "ranking_mean": float(np.mean(ranking_latencies)) if ranking_latencies else 0.0,
            "total_mean": float(np.mean(retrieval_latencies) + (np.mean(ranking_latencies) if ranking_latencies else 0.0)),
        },
        "catalog_size": num_items,
        "evaluation_users": len(eval_users),
    }

    print("\n--- STAGE 1: CANDIDATE GENERATION METRICS ---")
    cand_cov = catalog_coverage(all_cand_recs, num_items)
    summary["candidate_generation"]["catalog_coverage"] = round(cand_cov, 4)
    for k in k_values:
        r = float(np.mean(results["candidate_only"][f"recall@{k}"]))
        p = float(np.mean(results["candidate_only"][f"precision@{k}"]))
        n = float(np.mean(results["candidate_only"][f"ndcg@{k}"]))
        m = float(np.mean(results["candidate_only"][f"map@{k}"]))
        summary["candidate_generation"][f"recall@{k}"] = round(r, 4)
        summary["candidate_generation"][f"precision@{k}"] = round(p, 4)
        summary["candidate_generation"][f"ndcg@{k}"] = round(n, 4)
        summary["candidate_generation"][f"map@{k}"] = round(m, 4)
        print(f"  K={k:2d} | Recall: {r:.4f} | Precision: {p:.4f} | NDCG: {n:.4f} | MAP: {m:.4f}")
    print(f"  Catalog Coverage: {cand_cov * 100:.2f}% | Mean Retrieval Latency: {summary['latency_ms']['retrieval_mean']:.2f} ms")

    if has_ranker:
        print("\n--- STAGE 2: TWO-STAGE (CANDIDATE + RANKER) METRICS ---")
        ts_cov = catalog_coverage(all_twostage_recs, num_items)
        summary["two_stage_pipeline"]["catalog_coverage"] = round(ts_cov, 4)
        for k in k_values:
            r = float(np.mean(results["two_stage"][f"recall@{k}"]))
            p = float(np.mean(results["two_stage"][f"precision@{k}"]))
            n = float(np.mean(results["two_stage"][f"ndcg@{k}"]))
            m = float(np.mean(results["two_stage"][f"map@{k}"]))
            summary["two_stage_pipeline"][f"recall@{k}"] = round(r, 4)
            summary["two_stage_pipeline"][f"precision@{k}"] = round(p, 4)
            summary["two_stage_pipeline"][f"ndcg@{k}"] = round(n, 4)
            summary["two_stage_pipeline"][f"map@{k}"] = round(m, 4)
            print(f"  K={k:2d} | Recall: {r:.4f} | Precision: {p:.4f} | NDCG: {n:.4f} | MAP: {m:.4f}")
        print(f"  Catalog Coverage: {ts_cov * 100:.2f}% | Mean Ranking Latency: {summary['latency_ms']['ranking_mean']:.2f} ms")

    # Save to artifacts
    with open(eval_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepRec recommendation models.")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--candidate_k", type=int, default=100)
    parser.add_argument("--max_users", type=int, default=None)
    args = parser.parse_args()

    evaluate_system(
        config_path=args.config,
        candidate_k=args.candidate_k,
        max_eval_users=args.max_users,
    )


if __name__ == "__main__":
    main()
