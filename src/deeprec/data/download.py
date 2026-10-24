"""Download and extract the MovieLens 100K dataset."""

import argparse
from pathlib import Path
import urllib.request
import zipfile
import sys
from src.deeprec.config import load_config


REQUIRED_FILES = ["u.data", "u.item", "u.user", "u.genre"]


def download_movielens(
    raw_dir: str | Path = "data/raw/ml-100k",
    zip_path: str | Path = "data/raw/ml-100k.zip",
    url: str = "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    force: bool = False,
) -> bool:
    """Download and extract MovieLens 100K dataset if not already present.
    
    Args:
        raw_dir: Destination directory where u.data, u.item, etc. will reside.
        zip_path: Local path for the downloaded zip file.
        url: URL of the MovieLens 100K zip archive.
        force: If True, re-downloads even if files already exist.
        
    Returns:
        True if files are verified and present, False otherwise.
    """
    raw_dir = Path(raw_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if raw files already exist and are valid
    if not force and raw_dir.exists():
        missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
        if not missing:
            print(f"[DeepRec] Dataset already present in {raw_dir}. All required files verified.")
            return True
        else:
            print(f"[DeepRec] Missing files in {raw_dir}: {missing}. Will proceed with extraction/download.")

    # Check if zip exists
    if not zip_path.exists() or force:
        print(f"[DeepRec] Downloading MovieLens 100K from {url}...")
        try:
            # Custom user-agent to prevent blocking
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DeepRec/1.0")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, zip_path)
            print(f"[DeepRec] Download complete: {zip_path} ({zip_path.stat().st_size} bytes)")
        except Exception as e:
            print(f"[DeepRec] ERROR: Failed to download dataset from {url}: {e}", file=sys.stderr)
            return False

    # Extract
    print(f"[DeepRec] Extracting {zip_path} into {zip_path.parent}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(zip_path.parent)
        print(f"[DeepRec] Extraction complete.")
    except Exception as e:
        print(f"[DeepRec] ERROR: Failed to extract zip archive: {e}", file=sys.stderr)
        return False

    # Verify extracted files
    missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
    if missing:
        print(f"[DeepRec] ERROR: Expected files missing after extraction: {missing}", file=sys.stderr)
        return False

    print(f"[DeepRec] Dataset verified successfully at {raw_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download MovieLens 100K dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    success = download_movielens(
        raw_dir=paths.get("raw_data_dir", "data/raw/ml-100k"),
        zip_path=paths.get("raw_zip_path", "data/raw/ml-100k.zip"),
        url=paths.get("dataset_url", "https://files.grouplens.org/datasets/movielens/ml-100k.zip"),
        force=args.force,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
