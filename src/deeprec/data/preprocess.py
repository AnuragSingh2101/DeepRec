"""Preprocess MovieLens dataset with strict temporal split and zero-indexed padding."""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from src.deeprec.config import load_config


GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def preprocess_data(
    raw_dir: str | Path = "data/raw/ml-100k",
    out_dir: str | Path = "data/processed",
    min_interactions: int = 5,
) -> dict:
    """Preprocess MovieLens 100K data into temporal splits with deterministic ID mappings.
    
    Item ID 0 is strictly reserved as the PADDING token.
    
    Args:
        raw_dir: Directory containing raw MovieLens files.
        out_dir: Directory to store processed splits and metadata.
        min_interactions: Minimum interactions required per user.
        
    Returns:
        Dictionary with preprocessing summary.
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_file = raw_path / "u.data"
    item_file = raw_path / "u.item"
    user_file = raw_path / "u.user"

    if not data_file.exists() or not item_file.exists():
        raise FileNotFoundError(f"Raw data files missing in {raw_path}. Run download first.")

    # 1. Load interactions
    ratings = pd.read_csv(
        data_file,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )

    # 2. Load movies catalogue
    movie_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + [f"genre_{g}" for g in GENRE_NAMES]
    movies = pd.read_csv(
        item_file,
        sep="|",
        encoding="latin-1",
        header=None,
        names=movie_cols,
        engine="python"
    )

    # Extract year from release_date if possible
    def extract_year(date_str):
        if pd.isna(date_str) or not isinstance(date_str, str):
            return 1990 # Default fallback
        parts = date_str.split("-")
        if len(parts) == 3 and parts[2].isdigit():
            return int(parts[2])
        return 1990

    movies["release_year"] = movies["release_date"].apply(extract_year)

    # 3. Load users
    users = None
    if user_file.exists():
        users = pd.read_csv(
            user_file,
            sep="|",
            header=None,
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            engine="python"
        )

    # 4. Filter users by minimum activity
    user_counts = ratings.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_interactions].index
    ratings = ratings[ratings["user_id"].isin(valid_users)].copy()

    # 5. Deterministic sorting chronologically per user
    # Break timestamp ties deterministically by movie_id
    ratings = ratings.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)

    # 6. Build deterministic ID mappings
    # CRITICAL: ID 0 is RESERVED FOR PADDING for item embeddings
    # User IDs also mapped 1-indexed (0 reserved for unknown/cold-start)
    unique_users = sorted(ratings["user_id"].unique())
    # All catalogue movies should be mappable, even if not yet rated
    catalogue_movie_ids = sorted(movies["movie_id"].unique())

    user_map = {int(u): int(i + 1) for i, u in enumerate(unique_users)}
    movie_map = {int(m): int(i + 1) for i, m in enumerate(catalogue_movie_ids)}

    # Reverse maps
    idx_to_user = {int(v): int(k) for k, v in user_map.items()}
    idx_to_movie = {int(v): int(k) for k, v in movie_map.items()}

    # Map interactions
    ratings["user_idx"] = ratings["user_id"].map(user_map)
    ratings["movie_idx"] = ratings["movie_id"].map(movie_map)
    movies["movie_idx"] = movies["movie_id"].map(movie_map)

    # 7. Temporal Train / Validation / Test split
    # For each user, reserve the final interaction as Test,
    # the second-to-last interaction as Validation,
    # and all previous interactions as Training.
    ratings["position"] = ratings.groupby("user_idx").cumcount()
    ratings["user_count"] = ratings.groupby("user_idx")["movie_idx"].transform("size")

    train_df = ratings[ratings["position"] < (ratings["user_count"] - 2)].copy()
    val_df = ratings[ratings["position"] == (ratings["user_count"] - 2)].copy()
    test_df = ratings[ratings["position"] == (ratings["user_count"] - 1)].copy()

    # Ensure no empty user history in training
    assert len(train_df) > 0, "Train split is empty!"
    assert len(val_df) == len(valid_users), f"Val split mismatch: {len(val_df)} vs {len(valid_users)}"
    assert len(test_df) == len(valid_users), f"Test split mismatch: {len(test_df)} vs {len(valid_users)}"

    # 8. User metadata processing if present
    if users is not None:
        users = users[users["user_id"].isin(valid_users)].copy()
        users["user_idx"] = users["user_id"].map(user_map)
        # Categorical codes
        users["gender_code"] = (users["gender"] == "M").astype(int)
        occupations = sorted(users["occupation"].unique())
        occ_map = {occ: i for i, occ in enumerate(occupations)}
        users["occupation_code"] = users["occupation"].map(occ_map)
        users = users.sort_values("user_idx").reset_index(drop=True)
        users.to_csv(out_path / "users.csv", index=False)
        with open(out_path / "occupation_map.json", "w") as f:
            json.dump(occ_map, f, indent=2)

    # 9. Save CSV splits
    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "validation.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)
    movies.to_csv(out_path / "movies.csv", index=False)

    # 10. Save mappings and metadata
    with open(out_path / "user_map.json", "w") as f:
        json.dump(user_map, f)
    with open(out_path / "movie_map.json", "w") as f:
        json.dump(movie_map, f)

    meta = {
        "num_users": len(user_map),
        "num_movies": len(movie_map), # Total catalogue items
        "padding_idx": 0,
        "train_interactions": len(train_df),
        "val_interactions": len(val_df),
        "test_interactions": len(test_df),
        "min_interactions": min_interactions,
        "genres": GENRE_NAMES
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("         DEEPREC: DATA PREPROCESSING COMPLETED")
    print("=" * 60)
    print(f"Users mapped: {len(user_map):,}")
    print(f"Catalogue movies mapped: {len(movie_map):,} (ID 0 reserved for PADDING)")
    print(f"Train split interactions: {len(train_df):,}")
    print(f"Validation interactions:  {len(val_df):,}")
    print(f"Test interactions:        {len(test_df):,}")
    print(f"Saved processed datasets to: {out_path}")
    print("=" * 60)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens dataset into temporal splits.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})

    preprocess_data(
        raw_dir=paths.get("raw_data_dir", "data/raw/ml-100k"),
        out_dir=paths.get("processed_data_dir", "data/processed"),
        min_interactions=data_cfg.get("min_interactions_per_user", 5)
    )


if __name__ == "__main__":
    main()
