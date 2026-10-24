"""Training pipeline for the Deep Ranking Model (Stage 2)."""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

from src.deeprec.config import load_config, get_device, ensure_directories
from src.deeprec.data.ranking_features import RankingFeatureExtractor
from src.deeprec.models.ranking import DeepRankingModel
from src.deeprec.models.candidate_generation import CandidateGenerator


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ranking_dataset(
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    feature_extractor: RankingFeatureExtractor,
    candidate_model: Optional[CandidateGenerator] = None,
    device: torch.device = torch.device("cpu"),
    relevance_threshold: float = 4.0,
    neg_ratio: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct candidate-level training examples without target leakage.
    
    Positive examples: Rated movies with rating >= relevance_threshold (Label = 1).
    Negative examples: Rated movies with rating < relevance_threshold (Label = 0),
                       plus unchosen candidate items sampled per user (Label = 0).
                       
    Returns:
        user_ids, movie_ids, features_matrix, labels
    """
    user_list = []
    movie_list = []
    raw_feats_list = []
    label_list = []

    # Pre-index train interactions by user
    user_seen_movies = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    all_movies = list(feature_extractor.item_popularity.keys())

    # Build examples from observed ratings
    for _, row in train_df.iterrows():
        u = int(row["user_idx"])
        m = int(row["movie_idx"])
        r = float(row["rating"])
        ts = float(row["timestamp"])

        label = 1.0 if r >= relevance_threshold else 0.0

        # Candidate score: If candidate model is available, score it; else default 0.0
        cand_score = 0.0
        if candidate_model is not None:
            with torch.no_grad():
                u_vec = candidate_model.history_embedding(torch.tensor([m], device=device))
                cand_score = float(u_vec.norm().item())

        feat = feature_extractor.extract_raw_features(
            user_idx=u, movie_idx=m, candidate_score=cand_score, current_timestamp=ts
        )

        user_list.append(u)
        movie_list.append(m)
        raw_feats_list.append(feat)
        label_list.append(label)

    # Sample unchosen negative candidate impressions to balance positive / negative ratio
    np.random.seed(42)
    unique_users = train_df["user_idx"].unique()
    for u in unique_users:
        seen = user_seen_movies.get(u, set())
        # Sample neg_ratio unchosen movies
        candidates_to_sample = np.random.choice(all_movies, size=neg_ratio * 2, replace=False)
        neg_added = 0
        for m in candidates_to_sample:
            if m not in seen and m != 0:
                feat = feature_extractor.extract_raw_features(
                    user_idx=u, movie_idx=m, candidate_score=0.0
                )
                user_list.append(u)
                movie_list.append(m)
                raw_feats_list.append(feat)
                label_list.append(0.0)
                neg_added += 1
                if neg_added >= neg_ratio:
                    break

    raw_matrix = np.array(raw_feats_list, dtype=np.float32)

    # Fit feature normalization scaler on train features
    feature_extractor.fit_scaler(raw_matrix)
    transformed_features = feature_extractor.transform_features(raw_matrix)

    return (
        np.array(user_list, dtype=np.int64),
        np.array(movie_list, dtype=np.int64),
        transformed_features,
        np.array(label_list, dtype=np.float32),
    )


def train_ranking_model(
    config_path: str = "configs/model.yaml",
    epochs_override: int = None,
    batch_size_override: int = None,
) -> dict:
    start_time = time.time()
    cfg = load_config(config_path)
    ensure_directories(cfg)

    seed = cfg.get("system", {}).get("seed", 42)
    set_seed(seed)

    device = get_device(cfg.get("system", {}).get("device", "auto"))

    paths = cfg["paths"]
    data_dir = Path(paths["processed_data_dir"])
    artifact_dir = Path(paths["ranking_artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print(f"[DeepRec] Loading processed datasets for Ranking stage from {data_dir}...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    movies_df = pd.read_csv(data_dir / "movies.csv")
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    num_users = meta["num_users"]
    num_movies = meta["num_movies"]

    # 2. Fit Feature Extractor on training interactions ONLY
    print("[DeepRec] Fitting RankingFeatureExtractor strictly on training interactions...")
    feature_extractor = RankingFeatureExtractor(genres=meta.get("genres"))
    feature_extractor.fit(train_df=train_df, movies_df=movies_df)

    # 3. Construct training dataset
    data_cfg = cfg.get("data", {})
    relevance_th = data_cfg.get("relevance_rating_threshold", 4.0)

    print(f"[DeepRec] Constructing candidate-level ranking examples (relevance threshold >= {relevance_th})...")
    u_train, m_train, f_train, y_train = build_ranking_dataset(
        train_df=train_df,
        movies_df=movies_df,
        feature_extractor=feature_extractor,
        relevance_threshold=relevance_th,
        neg_ratio=2,
    )

    # 4. Construct validation dataset
    val_raw = []
    val_users = []
    val_movies = []
    val_labels = []
    for _, row in val_df.iterrows():
        u = int(row["user_idx"])
        m = int(row["movie_idx"])
        r = float(row["rating"])
        ts = float(row["timestamp"])
        feat = feature_extractor.extract_raw_features(
            user_idx=u, movie_idx=m, candidate_score=0.0, current_timestamp=ts
        )
        val_raw.append(feat)
        val_users.append(u)
        val_movies.append(m)
        val_labels.append(1.0 if r >= relevance_th else 0.0)

    f_val = feature_extractor.transform_features(np.array(val_raw, dtype=np.float32))
    u_val = np.array(val_users, dtype=np.int64)
    m_val = np.array(val_movies, dtype=np.int64)
    y_val = np.array(val_labels, dtype=np.float32)

    continuous_dim = f_train.shape[1]
    print(f"[DeepRec] Ranking dataset prepared: {len(y_train):,} train samples, {len(y_val):,} val samples.")
    print(f"[DeepRec] Continuous feature dimensions (after [x, x^2, sqrt(x)]): {continuous_dim}")

    # 5. DataLoaders
    rank_cfg = cfg.get("ranking", {})
    batch_size = batch_size_override or rank_cfg.get("batch_size", 256)
    epochs = epochs_override or rank_cfg.get("epochs", 12)
    patience = rank_cfg.get("early_stopping_patience", 3)

    train_ds = TensorDataset(
        torch.from_numpy(u_train).long(),
        torch.from_numpy(m_train).long(),
        torch.from_numpy(f_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 6. Initialize Deep Ranking Model
    model = DeepRankingModel(
        num_users=num_users,
        num_movies=num_movies,
        continuous_dim=continuous_dim,
        embedding_dim=rank_cfg.get("embedding_dim", 32),
        hidden_dims=rank_cfg.get("hidden_dims", [128, 64, 32]),
        dropout=rank_cfg.get("dropout", 0.1),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=rank_cfg.get("learning_rate", 0.001),
        weight_decay=rank_cfg.get("weight_decay", 1e-5),
    )

    print("=" * 60)
    print("          TRAINING DEEP RANKING MODEL (STAGE 2)")
    print("=" * 60)
    print(f"Device: {device} | Batches/epoch: {len(train_loader)} | Epochs: {epochs}")
    print(f"Input features: Categorical(users={num_users}, movies={num_movies}) + {continuous_dim} Continuous")
    print("=" * 60)

    best_val_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history_records = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for b_u, b_m, b_f, b_y in train_loader:
            b_u, b_m, b_f, b_y = b_u.to(device), b_m.to(device), b_f.to(device), b_y.to(device)

            optimizer.zero_grad()
            logits = model(b_u, b_m, b_f)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)

        # Validation evaluation: ROC-AUC and validation loss
        model.eval()
        with torch.no_grad():
            v_u = torch.from_numpy(u_val).long().to(device)
            v_m = torch.from_numpy(m_val).long().to(device)
            v_f = torch.from_numpy(f_val).float().to(device)
            v_y = torch.from_numpy(y_val).float().to(device)

            val_logits = model(v_u, v_m, v_f)
            val_loss = criterion(val_logits, v_y).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

            # ROC-AUC if both classes present
            if len(np.unique(y_val)) > 1:
                val_auc = float(roc_auc_score(y_val, val_probs))
            else:
                val_auc = 0.5

        record = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_auc": round(val_auc, 4),
        }
        history_records.append(record)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint_path = artifact_dir / "ranking_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_users": num_users,
                "num_movies": num_movies,
                "continuous_dim": continuous_dim,
                "embedding_dim": rank_cfg.get("embedding_dim", 32),
                "hidden_dims": rank_cfg.get("hidden_dims", [128, 64, 32]),
                "val_auc": val_auc,
                "val_loss": val_loss,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[DeepRec] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch} with Val AUC={best_val_auc:.4f})")
                break

    duration = time.time() - start_time

    # Save feature metadata & scaler params for inference
    ranking_metadata = {
        "model": "DeepRankingModel",
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "training_duration_seconds": round(duration, 2),
        "continuous_dim": continuous_dim,
        "feature_names": feature_extractor.get_feature_names(),
        "relevance_threshold": relevance_th,
        "history": history_records,
    }
    with open(artifact_dir / "ranking_metadata.json", "w") as f:
        json.dump(ranking_metadata, f, indent=2)

    # Save extractor attributes
    import pickle
    with open(artifact_dir / "feature_extractor.pkl", "wb") as f:
        pickle.dump(feature_extractor, f)

    print("=" * 60)
    print(f"Deep Ranking training complete in {duration:.1f}s.")
    print(f"Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}.")
    print(f"Saved ranking checkpoint to: {artifact_dir / 'ranking_model.pt'}")
    print("=" * 60)

    return ranking_metadata


def main():
    parser = argparse.ArgumentParser(description="Train Deep Ranking Model.")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override.")
    args = parser.parse_args()

    train_ranking_model(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
    )


if __name__ == "__main__":
    main()
