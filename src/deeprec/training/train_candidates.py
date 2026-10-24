"""Candidate generation model training script."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from src.deeprec.config import load_config, get_device, ensure_directories
from src.deeprec.data.sequences import build_rolling_sequences, build_eval_sequences
from src.deeprec.data.dataset import create_candidate_dataloaders
from src.deeprec.models.candidate_generation import CandidateGenerator
from src.deeprec.training.losses import SampledSoftmaxLoss
from src.deeprec.retrieval.retrieve import CandidateRetriever


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_negatives(
    positives: torch.Tensor,
    num_items: int,
    num_negatives: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample negative items uniformly from [1, num_items] excluding the target item."""
    batch_size = positives.shape[0]
    # Sample candidate negatives in [1, num_items]
    negatives = torch.randint(
        1, num_items + 1, (batch_size, num_negatives), dtype=torch.long, device=device
    )
    # If any negative equals the positive target, replace it
    mask = negatives == positives.unsqueeze(1)
    if mask.any():
        replacements = (negatives + 1) % num_items + 1
        negatives = torch.where(mask, replacements, negatives)

    return negatives


def evaluate_candidate_validation(
    retriever: CandidateRetriever,
    val_histories: np.ndarray,
    val_targets: np.ndarray,
    k_eval: int = 50,
) -> float:
    """Compute Recall@K on validation targets."""
    hits = 0
    total = len(val_targets)
    if total == 0:
        return 0.0

    batch_size = 256
    val_hist_tensor = torch.from_numpy(val_histories).long()

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_hist = val_hist_tensor[start_idx:end_idx]
        batch_targets = val_targets[start_idx:end_idx]

        indices, _ = retriever.retrieve_batch(batch_hist, k=k_eval, exclude_seen=True)
        top_cands = indices.cpu().numpy()

        for i, tgt in enumerate(batch_targets):
            if tgt in top_cands[i]:
                hits += 1

    return hits / total


def train_candidate_model(
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
    artifact_dir = Path(paths["candidate_artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load processed data
    print(f"[DeepRec] Loading processed datasets from {data_dir}...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    num_items = meta["num_movies"]
    user_features_df = None
    if (data_dir / "users.csv").exists():
        user_features_df = pd.read_csv(data_dir / "users.csv")

    # 2. Build sequences
    data_cfg = cfg.get("data", {})
    max_len = data_cfg.get("max_history_length", 20)

    print(f"[DeepRec] Generating rolling history sequences (max_len={max_len})...")
    train_hist, train_tgt, train_u = build_rolling_sequences(
        train_df, max_history_length=max_len, padding_idx=0
    )
    val_hist, val_tgt, val_u = build_eval_sequences(
        train_df, val_df, max_history_length=max_len, padding_idx=0
    )
    print(f"[DeepRec] Created {len(train_hist):,} train sequences and {len(val_hist):,} validation sequences.")

    # 3. Create DataLoaders
    cand_cfg = cfg.get("candidate_generation", {})
    batch_size = batch_size_override or cand_cfg.get("batch_size", 256)
    epochs = epochs_override or cand_cfg.get("epochs", 15)
    num_negatives = cand_cfg.get("num_negatives", 10)
    patience = cand_cfg.get("early_stopping_patience", 3)

    loaders = create_candidate_dataloaders(
        train_histories=train_hist,
        train_targets=train_tgt,
        train_users=train_u,
        user_features_df=user_features_df,
        batch_size=batch_size,
    )
    train_loader = loaders["train"]

    # 4. Initialize Candidate Generator Model
    use_context = cand_cfg.get("use_context_features", True) and (user_features_df is not None)
    model = CandidateGenerator(
        num_items=num_items,
        embedding_dim=cand_cfg.get("embedding_dim", 64),
        hidden_dims=cand_cfg.get("hidden_dims", [128, 64]),
        use_context=use_context,
        dropout=cand_cfg.get("dropout", 0.1),
    ).to(device)

    criterion = SampledSoftmaxLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cand_cfg.get("learning_rate", 0.001),
        weight_decay=cand_cfg.get("weight_decay", 1e-5),
    )

    # Precompute popular items for fallback
    popular_series = train_df["movie_idx"].value_counts()
    popular_items = popular_series.index.tolist()

    retriever = CandidateRetriever(
        model=model,
        num_items=num_items,
        device=device,
        popular_items=popular_items,
    )

    print("=" * 60)
    print("      TRAINING CANDIDATE GENERATION MODEL (STAGE 1)")
    print("=" * 60)
    print(f"Device: {device} | Batches/epoch: {len(train_loader)} | Epochs: {epochs}")
    print(f"Embedding Dim: {model.embedding_dim} | Context Features: {use_context}")
    print("=" * 60)

    best_val_recall = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history_records = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            histories = batch["history"].to(device)
            positives = batch["target"].to(device)

            context = None
            if use_context:
                context = {
                    "age": batch["age"].to(device),
                    "gender": batch["gender"].to(device),
                    "occupation": batch["occupation"].to(device),
                }

            optimizer.zero_grad()

            # Encode user representation
            user_vectors = model.encode(histories, context=context)

            # Sample negatives
            negatives = sample_negatives(positives, num_items, num_negatives, device)

            # Scores
            pos_scores = model.score(user_vectors, positives)
            neg_scores = model.score(user_vectors, negatives)

            loss = criterion(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        # Validation evaluation
        model.eval()
        val_recall_50 = evaluate_candidate_validation(retriever, val_hist, val_tgt, k_eval=50)

        record = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_recall_at_50": round(val_recall_50, 4),
        }
        history_records.append(record)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | Val Recall@50: {val_recall_50:.4f}")

        # Checkpointing
        if val_recall_50 > best_val_recall:
            best_val_recall = val_recall_50
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint_path = artifact_dir / "candidate_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_items": num_items,
                "embedding_dim": cand_cfg.get("embedding_dim", 64),
                "hidden_dims": cand_cfg.get("hidden_dims", [128, 64]),
                "use_context": use_context,
                "val_recall_at_50": val_recall_50,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[DeepRec] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch} with Val Recall@50={best_val_recall:.4f})")
                break

    duration = time.time() - start_time

    # Save training metadata
    train_summary = {
        "model": "CandidateGenerator",
        "best_epoch": best_epoch,
        "best_val_recall_at_50": best_val_recall,
        "training_duration_seconds": round(duration, 2),
        "device": str(device),
        "history": history_records,
        "popular_items": popular_items[:200],
        "num_items": num_items,
    }
    with open(artifact_dir / "training_summary.json", "w") as f:
        json.dump(train_summary, f, indent=2)

    print("=" * 60)
    print(f"Candidate Generation training complete in {duration:.1f}s.")
    print(f"Best Val Recall@50: {best_val_recall:.4f} at epoch {best_epoch}.")
    print(f"Saved checkpoint to: {artifact_dir / 'candidate_model.pt'}")
    print("=" * 60)

    return train_summary


def main():
    parser = argparse.ArgumentParser(description="Train Deep Candidate Generation model.")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override.")
    args = parser.parse_args()

    train_candidate_model(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
    )


if __name__ == "__main__":
    main()
