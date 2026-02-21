"""
-*- coding: utf-8 -*-
Name:        utils/process.py
Purpose:     Create train/val/test splits from datasets/sorted.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path
import random
import shutil
import csv
from typing import Iterable

from .config import IMG_EXTS, SplitConfig


def _list_images(folder: Path) -> list[Path]:
    """Return direct children whose suffix is in `IMG_EXTS`."""
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]


def _split_counts(n: int, ratios):
    """Convert ratios into integer (train, val, test) counts."""
    train = int(n * ratios[0])
    val = int(n * ratios[1])
    test = n - train - val
    return train, val, test


def _copy_many(files: Iterable[Path], dst: Path):
    """Copy files into `dst`."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)


def run_split(cfg: SplitConfig):
    """
    Create train/val/test directory splits from `cfg.root/sorted`.

    Workflow
    --------
    For each fake dataset:
    - shuffle images
    - allocate split counts from `cfg.ratios`
    - pair each fake with a unique real image
    - copy into: {root}/{split}/{dataset}/{0_real|1_fake}

    Real images are drawn globally without reuse across all datasets
    and splits.
    """

    random.seed(cfg.seed)

    sorted_root = cfg.root / "sorted"
    real_pool = _list_images(sorted_root / cfg.real_source)
    random.shuffle(real_pool)

    real_used = set()

    rows = []

    def take_real(n):
        selected = []
        for img in real_pool:
            if img not in real_used:
                selected.append(img)
                real_used.add(img)
            if len(selected) == n:
                break
        return selected

    def process_dataset(dataset_name: str, splits: dict):

        fake_files = _list_images(sorted_root / dataset_name)
        random.shuffle(fake_files)

        cursor = 0

        for split_name, count in splits.items():

            fake_subset = fake_files[cursor: cursor + count]
            cursor += count

            real_subset = take_real(len(fake_subset))

            fake_dst = cfg.root / split_name / dataset_name / "1_fake"
            real_dst = cfg.root / split_name / dataset_name / "0_real"

            _copy_many(fake_subset, fake_dst)
            _copy_many(real_subset, real_dst)

            for f in fake_subset:
                rows.append([split_name, dataset_name, "fake", f.name])

            for r in real_subset:
                rows.append([split_name, dataset_name, "real", r.name])

    # TRAIN / VAL
    for ds in cfg.trainval_sources:
        files = _list_images(sorted_root / ds)
        train_n, val_n, test_n = _split_counts(len(files), cfg.ratios)

        process_dataset(
            ds,
            {
                "train": train_n,
                "val": val_n,
            },
        )

    # TEST
    for ds in cfg.test_sources:
        files = _list_images(sorted_root / ds)
        _, _, test_n = _split_counts(len(files), cfg.ratios)

        process_dataset(
            ds,
            {
                "test": test_n,
            },
        )

    # CSV
    cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cfg.csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "dataset", "label", "filename"])
        writer.writerows(rows)
