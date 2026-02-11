"""
-*- coding: utf-8 -*-
Name:        utils/data.py
Purpose:     Dataset fetching utilities for DeepFail (strict + reproducible).
Author:      Kodama Chameleon <contact@kodamachameleon.com>
Co-Author:   ChatGPT 5.2
"""
from pathlib import Path
from typing import Iterable

from kaggle.api.kaggle_api_extended import KaggleApi

from .config import DatasetConfig, KaggleConfig


class DatasetDownloadError(RuntimeError):
    pass


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


# -------------------------
# Kaggle download helpers
# -------------------------

def _kaggle_auth(kcfg: KaggleConfig) -> KaggleApi:
    """
    Authenticate Kaggle using env vars (loaded from .env).
    Fails fast if auth fails.
    """
    # KaggleApi reads env vars internally; we still validate config for clarity.
    if not kcfg.username or not kcfg.key:
        raise DatasetDownloadError("KaggleConfig is missing username/key (should be impossible if make_config() used).")

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

    # Step-driven reproducibility: skip only if already present unless force=True.
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
        raise DatasetDownloadError(f"Failed to download Kaggle dataset '{slug}': {e}") from e

    return dest_dir


def fetch_sfhq_t2i(cfg: DatasetConfig, kcfg: KaggleConfig, *, force: bool = False) -> Path:
    return download_kaggle_dataset(cfg.sfhq_t2i_slug, cfg.sfhq_t2i_dir, kcfg=kcfg, unzip=True, force=force)


def fetch_tpdne(cfg: DatasetConfig, kcfg: KaggleConfig, *, force: bool = False) -> Path:
    return download_kaggle_dataset(cfg.tpdne_slug, cfg.tpdne_dir, kcfg=kcfg, unzip=True, force=force)


def fetch_ffhq(cfg: DatasetConfig, kcfg: KaggleConfig, *, force: bool = False) -> Path:
    return download_kaggle_dataset(cfg.ffhq_slug, cfg.ffhq_dir, kcfg=kcfg, unzip=True, force=force)


# -------------------------
# Convenience
# -------------------------

def fetch_all(
        cfg: DatasetConfig,
        kcfg: KaggleConfig,
        *,
        force: bool = False,
        include: Iterable[str] = ("sfhq_t2i", "tpdne", "ffhq"),
    ) -> dict[str, Path]:
    """
    Pull all datasets
    """
    results: dict[str, Path] = {}
    include_set = set(include)

    if "sfhq_t2i" in include_set:
        results["sfhq_t2i"] = fetch_sfhq_t2i(cfg, kcfg, force=force)
    if "tpdne" in include_set:
        results["tpdne"] = fetch_tpdne(cfg, kcfg, force=force)
    if "ffhq" in include_set:
        results["ffhq"] = fetch_ffhq(cfg, kcfg, force=force)

    return results
