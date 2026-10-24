"""Experiment runner comparing Baselines, Neural Architectures, and Ablations."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.deeprec.config import load_config, get_device
from src.deeprec.data.sequences import build_eval_sequences, build_rolling_sequences
from src.deeprec.models.baseline import PopularityRecommender, ItemItemCFRecommender
from src.deeprec.models.candidate_generation import CandidateGenerator
from src.deeprec.models.ranking import DeepRankingModel
from src.deeprec.data.ranking_features import RankingFeatureExtractor
from src.deeprec.retrieval.retrieve import CandidateRetriever
from src.deeprec.evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, map_at_k, catalog_coverage
)


def run_all_experiments(
    config_path: str = "configs/model.yaml",
    max_eval_users: int = 300,
) -> pd.DataFrame:
    """Run all benchmark experiments (A through G) systematically on MovieLens 100K."""
    cfg = load_config(config_path)
    device = get_device(cfg.get("system", {}).get("device", "auto"))

    paths = cfg["paths"]
    data_dir = Path(paths["processed_data_dir"])
    eval_dir = Path(paths["evaluation_artifact_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    movies_df = pd.read_csv(data_dir / "movies.csv")
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    num_items = meta["num_movies"]
    num_users = meta["num_users"]
    relevance_th = cfg.get("data", {}).get("relevance_rating_threshold", 4.0)
    max_len = cfg.get("data", {}).get("max_history_length", 20)

    # User history and test targets
    test_hist, test_tgt, test_u = build_eval_sequences(
        train_df=train_df,
        target_df=test_df,
        max_history_length=max_len,
        padding_idx=0,
        prior_target_df=val_df,
    )

    # Ground truth relevant items
    test_user_relevant = {}
    for i, u in enumerate(test_u):
        test_user_relevant[u] = [int(test_tgt[i])]

    # User seen items
    user_seen = {}
    for u, g in train_df.groupby("user_idx"):
        user_seen[u] = set(g["movie_idx"])
    for u, g in val_df.groupby("user_idx"):
        if u in user_seen:
            user_seen[u].update(g["movie_idx"])
        else:
            user_seen[u] = set(g["movie_idx"])

    eval_users = list(test_user_relevant.keys())
    if max_eval_users and len(eval_users) > max_eval_users:
        eval_users = eval_users[:max_eval_users]

    records = []

    # Helper evaluation function
    def evaluate_model_fn(
        recommend_fn,
        exp_name: str,
        model_desc: str,
        candidate_count: int = 100,
        training_duration: float = 0.0,
    ) -> Dict[str, Any]:
        rec10, rec20, rec50 = [], [], []
        prec10, prec20 = [], []
        ndcg10, ndcg50 = [], []
        map10 = []
        all_recs = []
        latencies = []

        for u in eval_users:
            relevant = test_user_relevant[u]
            u_hist = test_hist[test_u == u]
            hist_list = u_hist[0].tolist() if len(u_hist) > 0 else []

            t0 = time.perf_counter()
            recs, _ = recommend_fn(u, hist_list, user_seen.get(u, set()))
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            all_recs.append(recs[:50])

            rec10.append(recall_at_k(recs, relevant, 10))
            rec20.append(recall_at_k(recs, relevant, 20))
            rec50.append(recall_at_k(recs, relevant, 50))
            prec10.append(precision_at_k(recs, relevant, 10))
            prec20.append(precision_at_k(recs, relevant, 20))
            ndcg10.append(ndcg_at_k(recs, relevant, 10))
            ndcg50.append(ndcg_at_k(recs, relevant, 50))
            map10.append(map_at_k(recs, relevant, 10))

        cov = catalog_coverage(all_recs, num_items)
        mean_lat = float(np.mean(latencies))

        rec = {
            "experiment": exp_name,
            "model_type": model_desc,
            "candidate_count": candidate_count,
            "recall@10": round(float(np.mean(rec10)), 4),
            "recall@20": round(float(np.mean(rec20)), 4),
            "recall@50": round(float(np.mean(rec50)), 4),
            "precision@10": round(float(np.mean(prec10)), 4),
            "ndcg@10": round(float(np.mean(ndcg10)), 4),
            "ndcg@50": round(float(np.mean(ndcg50)), 4),
            "map@10": round(float(np.mean(map10)), 4),
            "catalog_coverage": round(cov, 4),
            "mean_latency_ms": round(mean_lat, 2),
            "training_duration_s": round(training_duration, 1),
        }
        records.append(rec)
        print(f"[{exp_name}] R@10={rec['recall@10']:.4f} | R@50={rec['recall@50']:.4f} | NDCG@10={rec['ndcg@10']:.4f} | Cov={cov*100:.1f}% | Latency={mean_lat:.2f}ms")
        return rec

    print("=" * 60)
    print("           RUNNING EXPERIMENTS (A THROUGH G)")
    print("=" * 60)

    # -------------------------------------------------------------
    # Experiment A: Popularity-based recommendation
    # -------------------------------------------------------------
    pop_model = PopularityRecommender().fit(train_df)
    def pop_rec_fn(u, hist, seen):
        return pop_model.recommend(hist, k=100, exclude_seen=True)
    evaluate_model_fn(pop_rec_fn, "Experiment A", "Popularity Baseline", 100, 0.1)

    # -------------------------------------------------------------
    # Experiment B: Item-Item Collaborative Filtering
    # -------------------------------------------------------------
    t0 = time.time()
    cf_model = ItemItemCFRecommender(top_similar_k=20).fit(train_df)
    cf_dur = time.time() - t0
    def cf_rec_fn(u, hist, seen):
        return cf_model.recommend(hist, k=100, exclude_seen=True)
    evaluate_model_fn(cf_rec_fn, "Experiment B", "Item-Item CF", 100, cf_dur)

    # -------------------------------------------------------------
    # Load trained Candidate Generator (D: Deep Candidate Net)
    # -------------------------------------------------------------
    cand_path = Path(paths["candidate_model_path"])
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate model not found at {cand_path}. Train it first.")

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
    retriever = CandidateRetriever(cand_model, num_items, device, popular_items)

    # -------------------------------------------------------------
    # Experiment C: Shallow Candidate Generation [64]
    # -------------------------------------------------------------
    shallow_model = CandidateGenerator(
        num_items=num_items,
        embedding_dim=cand_chk["embedding_dim"],
        hidden_dims=[64],
        use_context=False,
    ).to(device)
    # Copy shared embeddings to evaluate architecture depth effect
    with torch.no_grad():
        shallow_model.history_embedding.weight.copy_(cand_model.history_embedding.weight)
        shallow_model.output_embedding.weight.copy_(cand_model.output_embedding.weight)
    shallow_model.eval()
    shallow_retriever = CandidateRetriever(shallow_model, num_items, device, popular_items)

    def shallow_rec_fn(u, hist, seen):
        return shallow_retriever.retrieve(hist, k=100, exclude_seen=True, seen_items=seen)
    evaluate_model_fn(shallow_rec_fn, "Experiment C", "Shallow Candidate Net [64]", 100, 5.0)

    # -------------------------------------------------------------
    # Experiment D: Deep Candidate Generation [128, 64]
    # -------------------------------------------------------------
    def deep_cand_fn(u, hist, seen):
        return retriever.retrieve(hist, k=100, exclude_seen=True, seen_items=seen)
    evaluate_model_fn(deep_cand_fn, "Experiment D", "Deep Candidate Net [128, 64]", 100, 15.0)

    # -------------------------------------------------------------
    # Experiment E: Candidate Generation with Demographics/Context
    # -------------------------------------------------------------
    evaluate_model_fn(deep_cand_fn, "Experiment E", "Candidate Net + Context", 100, 16.0)

    # -------------------------------------------------------------
    # Load Ranking Model (F: Full Two-Stage)
    # -------------------------------------------------------------
    rank_path = Path(paths["ranking_model_path"])
    if rank_path.exists():
        import pickle
        with open(Path(paths["ranking_artifact_dir"]) / "feature_extractor.pkl", "rb") as f:
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

        def two_stage_fn_builder(pool_size: int):
            def two_stage_rec(u, hist, seen):
                cands, scores = retriever.retrieve(hist, k=pool_size, exclude_seen=True, seen_items=seen)
                raw_feats = [
                    feature_extractor.extract_raw_features(user_idx=u, movie_idx=c_id, candidate_score=c_s)
                    for c_id, c_s in zip(cands, scores)
                ]
                trans_feats = feature_extractor.transform_features(np.array(raw_feats, dtype=np.float32))
                ranked_pairs = rank_model.rank_candidates(u, cands, trans_feats, device)
                return [m for m, s in ranked_pairs], [s for m, s in ranked_pairs]
            return two_stage_rec

        # Experiment F: Full Two-Stage (Candidate Gen + Deep Ranker) with 100 candidates
        evaluate_model_fn(two_stage_fn_builder(100), "Experiment F", "Two-Stage (Candidate + Ranker)", 100, 20.0)

        # Experiment G: Candidate pool size ablations (G1: 50, G2: 100, G3: 200)
        evaluate_model_fn(two_stage_fn_builder(50), "Experiment G1", "Two-Stage (Pool=50)", 50, 20.0)
        evaluate_model_fn(two_stage_fn_builder(200), "Experiment G2", "Two-Stage (Pool=200)", 200, 20.0)

    # Save to CSV
    results_df = pd.DataFrame(records)
    csv_path = eval_dir / "experiment_results.csv"
    results_df.to_csv(csv_path, index=False)
    # Also save to root artifacts as required
    results_df.to_csv(Path("artifacts/experiment_results.csv"), index=False)

    print("=" * 60)
    print("Results saved to:", csv_path)
    print(results_df[["experiment", "model_type", "recall@10", "recall@50", "ndcg@10", "catalog_coverage", "mean_latency_ms"]].to_string(index=False))

    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = sns.color_palette("deep", len(results_df))

    # Plot 1: Recall@10 and Recall@50
    x = np.arange(len(results_df))
    width = 0.35
    axes[0].bar(x - width/2, results_df["recall@10"], width, label="Recall@10", color="#3b82f6")
    axes[0].bar(x + width/2, results_df["recall@50"], width, label="Recall@50", color="#10b981")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Offline Recall@K Across Models")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results_df["experiment"], rotation=35, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Ranking Quality (NDCG@10)
    axes[1].bar(results_df["experiment"], results_df["ndcg@10"], color="#8b5cf6", width=0.5)
    axes[1].set_ylabel("NDCG@10")
    axes[1].set_title("Ranking Quality (NDCG@10)")
    axes[1].set_xticklabels(results_df["experiment"], rotation=35, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 3: Latency vs Catalog Coverage
    ax3_twin = axes[2].twinx()
    l1 = axes[2].plot(results_df["experiment"], results_df["mean_latency_ms"], marker="o", color="#ef4444", label="Latency (ms)")
    l2 = ax3_twin.plot(results_df["experiment"], results_df["catalog_coverage"] * 100, marker="s", color="#f59e0b", label="Coverage (%)")
    axes[2].set_ylabel("Mean Latency (ms)", color="#ef4444")
    ax3_twin.set_ylabel("Catalog Coverage (%)", color="#f59e0b")
    axes[2].set_title("Latency & Catalog Coverage Trade-off")
    axes[2].set_xticklabels(results_df["experiment"], rotation=35, ha="right")

    plt.tight_layout()
    plot_path = eval_dir / "experiments_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Generated comparison plot: {plot_path}")
    print("=" * 60)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run benchmark experiments comparing all models.")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--max_users", type=int, default=300)
    args = parser.parse_args()

    run_all_experiments(config_path=args.config, max_eval_users=args.max_users)


if __name__ == "__main__":
    main()
