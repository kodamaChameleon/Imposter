"""
-*- coding: utf-8 -*-
Name:        utils/config.py
Purpose:     Configuration management for DeepFail.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

DOWNLOAD_CHOICES = ("all", "sfhq_t2i", "ffhq", "tdpne")

def load_env(dotenv_path: Path | None = None) -> None:
    """
    Load environment variables from .env if python-dotenv is installed.
    Safe to call multiple times.
    """
    if load_dotenv is None:
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)


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


def make_config(root: Path | str = "datasets", dotenv_path: Path | None = None) -> tuple[DatasetConfig, KaggleConfig]:
    """
    Build config after loading .env (if available).
    You can override DATA_ROOT via env, e.g. DATA_ROOT=/mnt/data/datasets
    """
    load_env(dotenv_path=dotenv_path)

    env_root = os.getenv("DATA_ROOT")
    final_root = Path(env_root) if env_root else Path(root)
    final_root = final_root.expanduser().resolve()

    return DatasetConfig(root=final_root), KaggleConfig()