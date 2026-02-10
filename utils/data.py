"""
-*- coding: utf-8 -*-
Name:        utils/data.py
Purpose:     Dataset handling utilities for DeepFail.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
Co-Author:   ChatGPT 5.2
"""
from pathlib import Path
from typing import Iterable, Sequence


SPLITS: tuple[str, ...] = ("train", "val", "test")
CLASSES: tuple[str, ...] = ("0_real", "1_fake")


def required_dataset_dirs(
        dataset_name: str,
        root: Path = Path("datasets"),
        splits: Sequence[str] = SPLITS,
    ) -> list[Path]:
    """
    Return the list of required directories for a given dataset.

    Structure:
      {root}/{train,val,test}/{dataset}/0_real
      {root}/{train,val,test}/{dataset}/1_fake
    """
    dirs: list[Path] = []
    for split in splits:
        for cls in CLASSES:
            dirs.append(root / split / dataset_name / cls)
    return dirs


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create directories if they don't exist (parents included)."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def provision_datasets(
        dataset_names: Iterable[str],
        root: Path = Path("datasets"),
        splits: Sequence[str] = SPLITS,
    ) -> list[Path]:
    """
    Ensure required directory structure exists for all dataset_names.

    Returns a flat list of all directories that are required (and now exist).
    """
    all_required: list[Path] = []
    for name in dataset_names:
        all_required.extend(required_dataset_dirs(name, root=root, splits=splits))

    ensure_dirs(all_required)
    return all_required
