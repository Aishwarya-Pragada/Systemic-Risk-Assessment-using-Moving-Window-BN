"""Configuration loading. All paths in the YAML are relative to the repository root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str((REPO_ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg
