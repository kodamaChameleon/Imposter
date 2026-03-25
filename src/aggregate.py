"""
-*- coding: utf-8 -*-
Name:        utils/aggregate.py
Purpose:     Build unified results dataset with detector + transforms.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MERGE_KEYS = ["base_dataset", "transform", "level"]

EXPECTED_COLUMNS = {
    "testset","accuracy","avg_precision","precision",
    "recall","f1","roc-auc","tn","fp","fn","tp"
}

ROUND_COLS = [
    "accuracy", "avg_precision", "precision",
    "recall", "f1", "roc-auc"
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _round_metrics(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    cols = [c for c in ROUND_COLS if c in df.columns]
    df[cols] = df[cols].round(decimals)
    return df


def _parse_dataset_name(name: str) -> dict:
    parts = name.split("-")

    if len(parts) == 1:
        return {
            "base_dataset": name,
            "transform": "none",
            "level": 100,
        }

    return {
        "base_dataset": parts[0],
        "transform": parts[1],
        "level": pd.to_numeric(parts[2], errors="coerce") if len(parts) > 2 else None,
    }


def _load_results(path: Path, detector: str) -> pd.DataFrame:
    """
    Load UFD / SAFE results with robust header detection.
    """

    # -----------------------------
    # locate header row
    # -----------------------------
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        cols = [c.strip().lower() for c in line.split(",")]
        if EXPECTED_COLUMNS.issubset(set(cols)):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"[aggregate] Header not found → {path}")

    # -----------------------------
    # read CSV
    # -----------------------------
    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [c.strip().lower() for c in df.columns]

    # -----------------------------
    # parse dataset structure
    # -----------------------------
    parsed = df["testset"].map(_parse_dataset_name).apply(pd.Series)
    parsed.columns = MERGE_KEYS

    df = pd.concat([df, parsed], axis=1)

    # -----------------------------
    # remove aggregate rows
    # -----------------------------
    df["base_dataset"] = df["base_dataset"].astype(str).str.strip()

    df = df[
        ~df["base_dataset"].str.lower().isin({
            "mean", "avg", "average"
        })
    ]

    # aggregate types BEFORE merge
    df["level"] = pd.to_numeric(df["level"], errors="coerce")

    # detector label
    df["detector"] = detector

    return df


def _load_transform_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(columns={"dataset": "base_dataset"})

    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["transform"] = df["transform"].fillna("none")

    return df[["base_dataset", "transform", "level", "lpips", "images"]]


def _attach_transforms(df: pd.DataFrame, transforms: pd.DataFrame) -> pd.DataFrame:
    """
    Merge LPIPS + images into results.
    """

    df = df.merge(
        transforms,
        on=MERGE_KEYS,
        how="left",
    )

    # originals → lpips = 0
    df["lpips"] = df["lpips"].fillna(0)

    # propagate image counts per dataset
    df["images"] = (
        df.groupby("base_dataset")["images"]
        .transform("max")
    )

    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final schema cleanup.
    """

    df = df.rename(columns={"base_dataset": "dataset"})

    # column ordering
    preferred = [
        "dataset", "transform", "level", "detector",
        "accuracy", "avg_precision", "precision",
        "recall", "f1", "roc-auc",
        "tn", "fp", "fn", "tp",
        "lpips", "images",
    ]

    cols = [c for c in preferred if c in df.columns]

    return df[cols].sort_values(
        ["dataset", "detector", "transform", "level"]
    )


def _build_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build weighted average rows across datasets.
    """

    group_cols = ["detector", "transform", "level"]

    metric_cols = [
        "accuracy", "avg_precision", "precision",
        "recall", "f1", "roc-auc", "lpips"
    ]

    count_cols = ["tn", "fp", "fn", "tp", "images"]

    def wmean(series, weights):
        mask = series.notna() & weights.notna()
        if not mask.any():
            return pd.NA
        return (series[mask] * weights[mask]).sum() / weights[mask].sum()

    grouped = (
        df.groupby(group_cols, dropna=False)
        .apply(
            lambda g: pd.Series({
                **{
                    col: wmean(g[col], g["images"])
                    for col in metric_cols
                },
                **{
                    col: g[col].sum(min_count=1)
                    for col in count_cols
                }
            })
        )
        .reset_index()
    )

    grouped["dataset"] = "average"

    return grouped


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_aggregation(input_paths, output_path):
    print("\n[aggregate] Loading inputs...")

    result_frames = []
    transform_frames = []

    # -----------------------------
    # ingest inputs
    # -----------------------------
    for key, typ, path in input_paths:
        print(f"[aggregate] → {path} ({key}, {typ})")

        if typ == "results":
            result_frames.append(_load_results(path, key))

        elif typ == "transform":
            transform_frames.append(_load_transform_metrics(path))

        else:
            raise ValueError(f"Unknown input type → {typ}")

    if not result_frames:
        raise ValueError("[aggregate] No result files provided")

    if not transform_frames:
        raise ValueError("[aggregate] No transform file provided")

    if len(transform_frames) > 1:
        raise ValueError("[aggregate] Multiple transform files provided")

    # -----------------------------
    # combine
    # -----------------------------
    df = pd.concat(result_frames, ignore_index=True)
    transforms = transform_frames[0]

    print("[aggregate] Attaching transforms...")
    df = _attach_transforms(df, transforms)

    print("[aggregate] Finalizing dataset...")
    df = _finalize(df)

    print("[aggregate] Building averages...")
    avg = _build_average(df)

    out = pd.concat([df, avg], ignore_index=True)

    out = _round_metrics(out, 2)
    if "images" in out.columns:
        out["images"] = (
            pd.to_numeric(out["images"], errors="coerce")
            .round()
            .astype("Int64")
        )
    out.to_csv(output_path, index=False)

    print(f"\n[ok] saved → {output_path}")
    print(f"[aggregate] rows: {len(out)}")

    return out
