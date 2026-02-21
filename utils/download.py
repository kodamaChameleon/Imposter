"""
-*- coding: utf-8 -*-
Name:        utils/data.py
Purpose:     Dataset fetching utilities for Imposter.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path
from typing import Iterable

from kaggle.api.kaggle_api_extended import KaggleApi

from .config import DatasetConfig, KaggleConfig


class DatasetDownloadError(RuntimeError):
    """Raised when Kaggle authentication or dataset download fails."""


def _ensure_dir(path: Path) -> None:
    """Create `path` (including parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _is_nonempty_dir(path: Path) -> bool:
    """Return True if `path` exists, is a directory, and contains ≥1 entry."""
    return path.exists() and path.is_dir() and any(path.iterdir())


def _kaggle_auth(kcfg: KaggleConfig) -> KaggleApi:
    """
    Authenticate Kaggle using env vars (loaded from .env).
    Fails fast if auth fails.
    """
    # KaggleApi reads env vars internally; validate config for clarity.
    if not kcfg.username or not kcfg.key:
        raise DatasetDownloadError("KaggleConfig is missing username/key \
            (should be impossible if make_config() used).")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise DatasetDownloadError(
            "Failed to authenticate Kaggle API. Verify:\n"
            "  - Your .env contains valid KAGGLE_USERNAME and KAGGLE_API_TOKEN\n"
            "  - The key is the Kaggle API token value (not your password)\n"
            f"Original error: {e}"
        ) from e
    return api


def download_kaggle_dataset(
    slug: str,
    dest_dir: Path,
    *,
    kcfg: KaggleConfig,
    unzip: bool = True,
    force: bool = False,
) -> Path:
    """
    Download a Kaggle dataset into dest_dir.
    Returns dest_dir.
    """
    _ensure_dir(dest_dir)

    # Skip only if already present unless force=True.
    if _is_nonempty_dir(dest_dir) and not force:
        return dest_dir

    api = _kaggle_auth(kcfg)

    try:
        api.dataset_download_files(
            dataset=slug,
            path=str(dest_dir),
            unzip=unzip,
            quiet=False,
            force=force,
        )
    except Exception as e:
        raise DatasetDownloadError(
            f"Failed to download Kaggle dataset '{slug}': {e}"
        ) from e

    return dest_dir


def fetch_dataset(
    dataset_name: str,
    cfg: DatasetConfig,
    kcfg: KaggleConfig,
    *,
    force: bool = False
) -> Path:
    """
    Generic downloader for a Kaggle dataset defined in DatasetConfig.

    dataset_name: 'sfhq_t2i', 'tpdne', or 'ffhq'
    """
    try:
        slug = getattr(cfg, f"{dataset_name}_slug")
        dest_dir = getattr(cfg, f"{dataset_name}_dir")
    except AttributeError as e:
        raise ValueError(
            f"DatasetConfig has no dataset named '{dataset_name}'"
        ) from e

    return download_kaggle_dataset(
        slug,
        dest_dir,
        kcfg=kcfg,
        unzip=True,
        force=force
    )


def fetch_all(
    cfg: DatasetConfig,
    kcfg: KaggleConfig,
    *,
    force: bool = False,
    include: Iterable[str] = ("sfhq_t2i", "tpdne", "ffhq"),
) -> dict[str, Path]:
    """
    Download multiple datasets.

    Parameters
    ----------
    include : Iterable[str] | Dataset names to fetch.
    """
    results: dict[str, Path] = {}
    for name in include:
        results[name] = fetch_dataset(name, cfg, kcfg, force=force)
    return results
