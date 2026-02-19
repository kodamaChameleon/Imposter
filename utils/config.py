"""
-*- coding: utf-8 -*-
Name:        utils/config.py
Purpose:     Configuration management for DeepFail.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

DOWNLOAD_CHOICES = ("all", "sfhq_t2i", "ffhq", "tdpne")
TRANSFORM_CHOICES = [
    "all", "compression", "resize",
    "crop", "contrast", "saturation"
]

def load_env(dotenv_path: Path | None = None) -> None:
    """
    Load environment variables from .env if python-dotenv is installed.
    Safe to call multiple times.
    """
    if load_dotenv is None:
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)

# -------------------------
# Env helpers
# -------------------------

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class KaggleConfig:
    """
    Kaggle expects KAGGLE_USERNAME and KAGGLE_KEY in env, or ~/.kaggle/kaggle.json.
    We prefer .env for local dev/CI.
    """
    username: str | None = os.getenv("KAGGLE_USERNAME")
    key: str | None = os.getenv("KAGGLE_API_TOKEN")


@dataclass(frozen=True)
class DatasetConfig:
    root: Path

    # Kaggle dataset identifiers
    sfhq_t2i_slug: str = "selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models"
    tpdne_slug: str = "almightyj/person-face-dataset-thispersondoesnotexist"
    ffhq_slug: str = "gibi13/flickr-faces-hq-dataset-ffhq"

    # Folder names under root
    sfhq_t2i_dirname: str = "SFHQ-T2I"
    tpdne_dirname: str = "TPDNE"
    ffhq_dirname: str = "FFHQ"

    @property
    def sfhq_t2i_dir(self) -> Path:
        return self.root / self.sfhq_t2i_dirname

    @property
    def tpdne_dir(self) -> Path:
        return self.root / self.tpdne_dirname

    @property
    def ffhq_dir(self) -> Path:
        return self.root / self.ffhq_dirname

    @property
    def ffhq_repo_dir(self) -> Path:
        """
        Where we clone NVLabs/ffhq-dataset to run download_ffhq.py (kept under ffhq_dir).
        """
        return self.ffhq_dir / "_ffhq-dataset-repo"


@dataclass(frozen=True)
class JpegConfig:
    """
    JPEG encoding settings used to standardize ALL images through the same lossy pipeline.
    """
    quality = _env_int("JPEG_QUALITY", 95)               # Pillow quality
    subsampling = _env_int("JPEG_SUBSAMPLING", 0)        # 4:4:4 (no chroma subsampling)
    optimize = _env_bool("JPEG_OPTIMIZE", True)          # better Huffman tables (slower encode, smaller file)
    progressive = _env_bool("JPEG_PROGRESSIVE", False)   # baseline JPEG for consistency/simplicity


def make_config(root: Path | str = "datasets", dotenv_path: Path | None = None) -> tuple[DatasetConfig, KaggleConfig, JpegConfig]:
    """
    Build config after loading .env (if available).
    You can override DATA_ROOT via env, e.g. DATA_ROOT=/mnt/data/datasets
    """
    load_env(dotenv_path=dotenv_path)

    env_root = os.getenv("DATA_ROOT")
    final_root = Path(env_root) if env_root else Path(root)
    final_root = final_root.expanduser().resolve()

    return DatasetConfig(root=final_root), KaggleConfig(), JpegConfig()