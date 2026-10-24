"""PyTorch Dataset implementations for candidate generation and ranking models."""

from typing import Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CandidateDataset(Dataset):
    """Dataset for training Candidate Generation neural network.
    
    Returns:
        history: LongTensor of shape [max_history_length]
        target: LongTensor scalar (positive item ID)
        user_idx: LongTensor scalar
        context: Dict with demographic context tensors (age, gender, occupation)
    """

    def __init__(
        self,
        histories: np.ndarray,
        targets: np.ndarray,
        user_ids: np.ndarray,
        user_features_df: Optional[pd.DataFrame] = None,
    ):
        self.histories = torch.from_numpy(histories).long()
        self.targets = torch.from_numpy(targets).long()
        self.user_ids = torch.from_numpy(user_ids).long()

        self.has_user_features = False
        if user_features_df is not None:
            # Map user_idx to demographic features
            max_user = int(user_ids.max()) if len(user_ids) > 0 else 0
            if "user_idx" in user_features_df.columns:
                max_df_user = int(user_features_df["user_idx"].max())
                max_user = max(max_user, max_df_user)

            # Pre-allocate lookup arrays for fast O(1) indexing
            self.age_lookup = np.zeros(max_user + 1, dtype=np.float32)
            self.gender_lookup = np.zeros(max_user + 1, dtype=np.int64)
            self.occ_lookup = np.zeros(max_user + 1, dtype=np.int64)

            for _, row in user_features_df.iterrows():
                uidx = int(row["user_idx"])
                if uidx <= max_user:
                    # Normalized age: (age - 34) / 15 approx
                    age = float(row.get("age", 30))
                    self.age_lookup[uidx] = (age - 34.0) / 15.0
                    self.gender_lookup[uidx] = int(row.get("gender_code", 0))
                    self.occ_lookup[uidx] = int(row.get("occupation_code", 0))

            self.has_user_features = True

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "history": self.histories[idx],
            "target": self.targets[idx],
            "user_idx": self.user_ids[idx],
        }
        if self.has_user_features:
            uidx = int(self.user_ids[idx].item())
            item["age"] = torch.tensor(self.age_lookup[uidx], dtype=torch.float32)
            item["gender"] = torch.tensor(self.gender_lookup[uidx], dtype=torch.long)
            item["occupation"] = torch.tensor(self.occ_lookup[uidx], dtype=torch.long)
        else:
            item["age"] = torch.tensor(0.0, dtype=torch.float32)
            item["gender"] = torch.tensor(0, dtype=torch.long)
            item["occupation"] = torch.tensor(0, dtype=torch.long)

        return item


def create_candidate_dataloaders(
    train_histories: np.ndarray,
    train_targets: np.ndarray,
    train_users: np.ndarray,
    val_histories: Optional[np.ndarray] = None,
    val_targets: Optional[np.ndarray] = None,
    val_users: Optional[np.ndarray] = None,
    user_features_df: Optional[pd.DataFrame] = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for training and validation."""
    train_ds = CandidateDataset(
        histories=train_histories,
        targets=train_targets,
        user_ids=train_users,
        user_features_df=user_features_df,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    loaders = {"train": train_loader}

    if val_histories is not None and val_targets is not None and val_users is not None:
        val_ds = CandidateDataset(
            histories=val_histories,
            targets=val_targets,
            user_ids=val_users,
            user_features_df=user_features_df,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        loaders["val"] = val_loader

    return loaders
