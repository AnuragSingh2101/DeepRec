"""Configuration loader and device resolution for DeepRec."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import torch


DEFAULT_CONFIG_PATH = Path("configs/model.yaml")


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to configuration file. If None, defaults to configs/model.yaml.
        
    Returns:
        Nested dictionary of configuration options.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        # Fallback to root relative search if called from a subdirectory
        alt_path = Path(__file__).resolve().parent.parent.parent / "configs" / "model.yaml"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Resolve compute device based on configuration and hardware availability.
    
    Args:
        requested_device: 'auto', 'cuda', or 'cpu'. If None, auto-detected.
        
    Returns:
        torch.device instance.
    """
    if requested_device is None or requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[DeepRec] Hardware acceleration enabled: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("[DeepRec] CUDA not detected or requested. Using CPU.")
    elif requested_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("[DeepRec] WARNING: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def ensure_directories(config: Dict[str, Any]) -> None:
    """Ensure that all artifact, data, and output directories exist."""
    paths = config.get("paths", {})
    for key, path_str in paths.items():
        if "dir" in key:
            p = Path(path_str)
            p.mkdir(parents=True, exist_ok=True)
