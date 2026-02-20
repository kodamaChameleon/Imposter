"""
-*- coding: utf-8 -*-
Name:        utils/config.py
Purpose:     Configuration management for DeepFail.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv


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
class DefaultOptions:
    """
    Define default options for argument parsing
    """
    root: Path = Path("datasets")
    download_choices: tuple[str, ...] = ("all", "sfhq_t2i", "ffhq", "tdpne")

    feature_model: str = "inception"
    feature_choices: list[str] = field(default_factory=lambda: ["inception", "dinov2_vitb14"])

    clip_mode: str = "sliding"
    clip_choices: list[str] = field(default_factory=lambda: ["sliding", "truncate"])

    trainval_set: list[str] = field(default_factory=lambda: ["FLUX1_dev"])

    test_set: list[str] = field(default_factory=lambda: [
        "FLUX1_dev", "FLUX1_pro", "FLUX1_schnell", "SDXL", "TPDNE"
    ])

    real_set: str = "FFHQ"

    split_ratios: list[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])

    split_csv: Path = root / "train_val_test.csv"
    split_seed: int = 1337

    transform_opt: list[str] = field(default_factory=lambda: ["all"])

    transform_choices: list[str] = field(default_factory=lambda: [
        "all", "compression", "resize", "crop", "contrast", "saturation"
    ])

    transform_level: list[int] = field(default_factory=lambda: [6, 10])

@dataclass(frozen=True)
class KaggleConfig:
    """
    Kaggle expects KAGGLE_USERNAME and KAGGLE_KEY in env, or ~/.kaggle/kaggle.json.
    Prefered .env for local dev/CI.
    """
    username: str | None = os.getenv("KAGGLE_USERNAME")
    key: str | None = os.getenv("KAGGLE_API_TOKEN")


@dataclass(frozen=True)
class DatasetConfig:
    """
    Dataset configs for downloading
    """
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
        """
        Local directory for SFHQ-T2I dataset
        """
        return self.root / self.sfhq_t2i_dirname

    @property
    def tpdne_dir(self) -> Path:
        """
        Local directory for This Person Does Not Exist dataset
        """
        return self.root / self.tpdne_dirname

    @property
    def ffhq_dir(self) -> Path:
        """
        Local directory for FFHQ dataset
        """
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
    optimize = _env_bool("JPEG_OPTIMIZE", True)          # slower encode, smaller file
    progressive = _env_bool("JPEG_PROGRESSIVE", False)   # baseline JPEG for consistency/simplicity


@dataclass(frozen=True)
class AppConfig:
    """
    Super config class
    """
    dataset: DatasetConfig
    kaggle: KaggleConfig
    jpeg: JpegConfig


def make_config(
        root: Path | str = "datasets",
        dotenv_path: Path | None = None,
    ) -> AppConfig:
    """
    Compile/Initialize Configs
    """
    load_env(dotenv_path)

    env_root = os.getenv("DATA_ROOT")
    final_root = Path(env_root) if env_root else Path(root)
    final_root = final_root.expanduser().resolve()

    dataset = DatasetConfig(root=final_root)
    kaggle = KaggleConfig()
    jpeg = JpegConfig()

    return AppConfig(
        dataset=dataset,
        kaggle=kaggle,
        jpeg=jpeg,
    )
