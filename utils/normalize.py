"""
-*- coding: utf-8 -*-
Name:        utils/normalize.py
Purpose:     Format and merge results data.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
MERGE_KEYS = ["base_dataset", "transform", "level"]


def _is_UFD_acc(text: str) -> bool:
    parts = text.split()

    try:
        float(parts[1])
        float(parts[2])
        float(parts[3])
        return True
    except ValueError:
        return False


def _parse_dataset_name(name: str) -> dict:
    """
    FLUX1_dev-compression-35 → 
    base=FLUX1_dev, transform=compression, level=35
    """
    parts = name.split("-")

    if len(parts) == 1:
        return {"base_dataset": name, "transform": None, "level": None}

    base = parts[0]
    transform = parts[1]
    level = parts[2] if len(parts) > 2 else None

    return {
        "base_dataset": base,
        "transform": transform,
        "level": pd.to_numeric(level, errors="coerce"),
    }


def _load_transform_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(columns={"dataset": "base_dataset"})
    df["level"] = pd.to_numeric(df["level"], errors="coerce")

    return df[["base_dataset", "transform", "level", "lpips", "images"]]


def _load_safe(path: Path) -> pd.DataFrame:
    # --- find header row ---
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "testset" in line.lower() and "accuracy" in line.lower():
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"[normalize] SAFE header not found → {path}")

    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [c.strip().lower() for c in df.columns]

    parsed = df["testset"].map(_parse_dataset_name).apply(pd.Series)
    parsed.columns = MERGE_KEYS

    out = pd.concat([parsed, df], axis=1)

    # Remove aggregate rows like "mean"
    out["base_dataset"] = out["base_dataset"].astype(str).str.strip()
    out = out[~out["base_dataset"].str.lower().isin({"mean", "avg", "average"})]

    acc = out.copy()
    acc["SAFE_accuracy"] = acc["accuracy"]

    prec = out.copy()
    prec["SAFE_precision"] = prec["avg precision"]

    return pd.merge(
        acc[
            ["base_dataset", "transform", "level", "SAFE_accuracy"]
        ],
        prec[
            ["base_dataset", "transform", "level", "SAFE_precision"]
        ],
        on=MERGE_KEYS,
        how="outer",
    )


def _load_ufd_accuracy(path: Path) -> pd.DataFrame:
    rows = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        name, values = line.split(":")
        _, _, acc_avg = map(float, values.split())

        parsed = _parse_dataset_name(name.strip())

        rows.append(
            {
                **parsed,
                "UFD_accuracy": acc_avg,
            }
        )

    return pd.DataFrame(rows)


def _load_ufd_precision(path: Path) -> pd.DataFrame:
    rows = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        name, value = line.split(":")
        parsed = _parse_dataset_name(name.strip())

        rows.append(
            {
                **parsed,
                "UFD_precision": float(value.strip()),
            }
        )

    return pd.DataFrame(rows)


def _detect_and_load(path: Path) -> pd.DataFrame:

    # Read head to determine file type
    with path.open() as f:
        head = f.read(2048).lower()

    if path.suffix == ".csv" and "lpips" in head:
        return _load_transform_metrics(path)

    if path.suffix == ".csv" and "avg precision" in head:
        return _load_safe(path)

    if path.suffix == ".txt" and _is_UFD_acc(head):
        return _load_ufd_accuracy(path)

    if path.suffix == ".txt":
        return _load_ufd_precision(path)

    raise ValueError(f"[normalize] Unknown file type → {path}")


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for df in frames[1:]:
        out = pd.merge(out, df, on=MERGE_KEYS, how="outer")
    return out


def _finalize_normalized(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()

    # Rename base_dataset → dataset
    df = df.rename(columns={"base_dataset": "dataset"})

    # Set lpips values to 0 for originals
    df["transform"] = df["transform"].fillna("none")
    df["lpips"] = df["lpips"].fillna(0)

    # Add image counts to originals
    if "images" in df.columns:
        df["images"] = (
            df.groupby("dataset")["images"]
            .transform("max")
        )
    
    # Add level to originals
    orig_mask = (df["transform"] == "none") & (df["level"].isna())
    df.loc[orig_mask, "level"] = 100

    # Images → integer
    if "images" in df.columns:
        df["images"] = pd.to_numeric(df["images"], errors="coerce").astype("Int64")

    # Column ordering
    preferred = [
        "dataset",
        "transform",
        "level",
        "UFD_accuracy",
        "UFD_precision",
        "SAFE_accuracy",
        "SAFE_precision",
        "lpips",
        "images",
    ]

    cols = [c for c in preferred if c in df.columns]
    df = df[cols]

    return df.sort_values(["dataset", "transform", "level"])


def _build_average_dataset(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    group_cols = ["transform", "level"]

    def wmean(series, weights):
        mask = series.notna() & weights.notna()
        if not mask.any():
            return pd.NA
        return (series[mask] * weights[mask]).sum() / weights[mask].sum()

    grouped = (
        work.groupby(group_cols, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "UFD_accuracy": wmean(g["UFD_accuracy"], g["images"]),
                    "UFD_precision": wmean(g["UFD_precision"], g["images"]),
                    "SAFE_accuracy": wmean(g["SAFE_accuracy"], g["images"]),
                    "SAFE_precision": wmean(g["SAFE_precision"], g["images"]),
                    "lpips": wmean(g["lpips"], g["images"]),
                    "images": g["images"].sum(min_count=1),
                }
            )
        )
        .reset_index()
    )

    grouped.insert(0, "dataset", "AVERAGE")

    return grouped


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_normalization(
    input_paths: list[Path],
    output_path: Path,
) -> pd.DataFrame:

    print("\n[normalize] Loading inputs…")

    frames = []

    for path in input_paths:
        print(f"[normalize] → {path}")
        frames.append(_detect_and_load(path))

    master = _merge_frames(frames)
    master = _finalize_normalized(master)

    master.to_csv(output_path, index=False)

    print(f"\n[ok] normalized CSV saved → {output_path}")
    print(f"[normalize] rows: {len(master)}")

    # build summary
    avg = _build_average_dataset(master)

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    avg.to_csv(summary_path, index=False)

    print(f"[ok] summary CSV saved → {summary_path}")

    return master
