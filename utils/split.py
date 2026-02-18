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
from dataclasses import dataclass
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class SplitConfig:
    root: Path
    trainval_sources: list[str]
    test_sources: list[str]
    real_source: str
    ratios: tuple[float, float, float]
    csv_path: Path
    seed: int = 1337


def _list_images(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def _split_counts(n: int, ratios):
    train = int(n * ratios[0])
    val = int(n * ratios[1])
    test = n - train - val
    return train, val, test


def _copy_many(files: Iterable[Path], dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)


def run_split(
    root: Path,
    trainval_sources: list[str],
    test_sources: list[str],
    real_source: str,
    ratios: tuple[float, float, float],
    csv_path: Path,
    seed: int = 1337,
):

    cfg = SplitConfig(
        root=root,
        trainval_sources=trainval_sources,
        test_sources=test_sources,
        real_source=real_source,
        ratios=ratios,
        csv_path=csv_path,
        seed=seed,
    )

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
