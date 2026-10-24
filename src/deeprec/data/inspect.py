"""Inspect raw MovieLens dataset shapes, columns, distributions, and missing values."""

import argparse
from pathlib import Path
import pandas as pd
from src.deeprec.config import load_config


GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def inspect_dataset(raw_dir: str | Path = "data/raw/ml-100k") -> dict:
    """Inspect dataset and return summary metrics.
    
    Args:
        raw_dir: Directory containing raw MovieLens files.
        
    Returns:
        Dictionary with inspection statistics.
    """
    raw_path = Path(raw_dir)
    data_file = raw_path / "u.data"
    item_file = raw_path / "u.item"
    user_file = raw_path / "u.user"

    if not data_file.exists() or not item_file.exists():
        raise FileNotFoundError(f"Raw data files not found in {raw_path}. Run download first.")

    # 1. Load ratings
    ratings = pd.read_csv(
        data_file,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )

    # 2. Load movies
    movie_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + [f"genre_{g}" for g in GENRE_NAMES]
    movies = pd.read_csv(
        item_file,
        sep="|",
        encoding="latin-1",
        header=None,
        names=movie_cols,
        engine="python"
    )

    # 3. Load users if present
    users = None
    if user_file.exists():
        users = pd.read_csv(
            user_file,
            sep="|",
            header=None,
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            engine="python"
        )

    print("=" * 60)
    print("           DEEPREC: MOVIELENS 100K DATASET INSPECTION")
    print("=" * 60)

    num_ratings = len(ratings)
    num_users = ratings["user_id"].nunique()
    num_movies = ratings["movie_id"].nunique()
    total_catalogue_movies = len(movies)
    sparsity = 1.0 - (num_ratings / (num_users * total_catalogue_movies))

    print(f"\n[1] Overall Shapes & Sparsity:")
    print(f"  - Total interactions (ratings): {num_ratings:,}")
    print(f"  - Unique users with interactions: {num_users:,}")
    print(f"  - Unique rated movies: {num_movies:,}")
    print(f"  - Total movies in catalogue: {total_catalogue_movies:,}")
    print(f"  - Interaction matrix sparsity: {sparsity * 100:.2f}%")

    print(f"\n[2] Rating Distribution:")
    rating_counts = ratings["rating"].value_counts().sort_index()
    for r, count in rating_counts.items():
        pct = (count / num_ratings) * 100
        bar = "#" * int(pct // 2)
        print(f"  Rating {r}: {count:6d} ({pct:5.1f}%) | {bar}")

    print(f"\n[3] User Activity Statistics:")
    user_counts = ratings.groupby("user_id").size()
    print(f"  - Min interactions/user: {user_counts.min()}")
    print(f"  - Median interactions/user: {user_counts.median():.1f}")
    print(f"  - Mean interactions/user: {user_counts.mean():.1f}")
    print(f"  - Max interactions/user: {user_counts.max()}")

    print(f"\n[4] Missing Values Check:")
    print(f"  - Ratings null count: {ratings.isnull().sum().to_dict()}")
    print(f"  - Movies null count: {movies[['movie_id', 'title', 'release_date']].isnull().sum().to_dict()}")
    if users is not None:
        print(f"  - Users null count: {users.isnull().sum().to_dict()}")

    print(f"\n[5] Sample Ratings:")
    print(ratings.head(3).to_string(index=False))

    print(f"\n[6] Sample Movie Metadata:")
    print(movies[["movie_id", "title", "release_date"] + [f"genre_{g}" for g in GENRE_NAMES[:3]]].head(3).to_string(index=False))

    print("=" * 60)

    summary = {
        "num_ratings": num_ratings,
        "num_users": num_users,
        "num_movies": num_movies,
        "total_catalogue_movies": total_catalogue_movies,
        "sparsity": sparsity,
        "rating_distribution": rating_counts.to_dict(),
        "min_interactions_per_user": int(user_counts.min()),
        "max_interactions_per_user": int(user_counts.max()),
        "mean_interactions_per_user": float(user_counts.mean())
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect MovieLens dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg.get("paths", {}).get("raw_data_dir", "data/raw/ml-100k")
    inspect_dataset(raw_dir)


if __name__ == "__main__":
    main()
