"""
-*- coding: utf-8 -*-
Name:        utils/graph.py
Purpose:     Generate tables/graphs from result CSV files.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data container (matches style of kid/clip result objects)
# ---------------------------------------------------------------------
MERGE_KEYS = ["base_dataset", "transform", "level"]

@dataclass
class ClipGraphRow:
    dataset: str
    count: int
    score_truncate: float | None
    score_sliding: float | None


@dataclass
class KidGraphRow:
    dataset: str
    count: int
    score_inception: float | None
    score_dinov2: float | None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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

    # Null handling
    df["transform"] = df["transform"].fillna("none")
    df["lpips"] = df["lpips"].fillna(0)

    if "images" in df.columns:
        df["images"] = (
            df.groupby("dataset")["images"]
            .transform("max")
        )

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


# ---------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------

def _build_clip_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    pivot = df.pivot(index="dataset", columns="clip_mode", values="score")

    counts = df.groupby("dataset")["num_images"].first()

    table = pd.DataFrame(index=pivot.index)
    table["count"] = counts

    if "truncate" in pivot.columns:
        table["score (truncate)"] = pivot["truncate"]

    if "sliding" in pivot.columns:
        table["score (sliding)"] = pivot["sliding"]

    return table.sort_index()


def _build_kid_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"[graph] CSV is empty → {csv_path}")

    # Use only basename of dataset_b
    df["dataset"] = Path("").joinpath  # placeholder to keep lint happy
    df["dataset"] = df["dataset_b"].map(lambda p: Path(p).name)

    pivot = df.pivot(index="dataset", columns="feature_model", values="kid_mean")

    counts = (
        df.groupby("dataset")["num_images_b"]
        .first()
        .reindex(pivot.index)
    )

    table = pd.DataFrame(index=pivot.index)
    table["count"] = counts

    if "inception" in pivot.columns:
        table["score (inception)"] = pivot["inception"]

    if "dinov2_vitb14" in pivot.columns:
        table["score (dinov2_vitb14)"] = pivot["dinov2_vitb14"]

    return table.sort_index()


def _save_table_png(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    display_df = df.copy()

    # Formatting
    display_df["count"] = display_df["count"].map(lambda x: f"{int(x):,}")

    for col in display_df.columns:
        if col.startswith("score"):
            display_df[col] = display_df[col].map(
                lambda x: "—" if pd.isna(x) else f"{x:.4f}"
            )

    fig_height = 0.40 * len(display_df) 
    fig_width = len(display_df.columns)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        rowLabels=display_df.index,
        loc="center",
        cellLoc="right",
        colLoc="right",
    )

    table.auto_set_font_size(False)
    table.auto_set_column_width(col=list(range(len(display_df.columns))))
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAF2")

    # Zebra striping
    for i in range(1, len(display_df) + 1):
        if i % 2 == 0:
            for j in range(len(display_df.columns)):
                table[(i, j)].set_facecolor("#F7F7F7")

    ax.set_title(title, fontsize=14, weight="bold", pad=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def run_graph_clip(csv_path: Path) -> list[ClipGraphRow]:
    table = _build_clip_table(csv_path)
    title = "CLIP Score Comparison"

    print(f"\n[graph] {title}\n")
    print(
        table.to_string(
            float_format=lambda x: "—" if pd.isna(x) else f"{x:.4f}"
        )
    )

    output_path = csv_path.with_suffix(".png")
    _save_table_png(table, output_path, title)

    print(f"\n[ok] graph saved → {output_path}")

    rows: list[ClipGraphRow] = []

    for dataset, r in table.iterrows():
        rows.append(
            ClipGraphRow(
                dataset=dataset,
                count=int(r["count"]),
                score_truncate=None
                if "score (truncate)" not in table.columns or pd.isna(r.get("score (truncate)"))
                else float(r["score (truncate)"]),
                score_sliding=None
                if "score (sliding)" not in table.columns or pd.isna(r.get("score (sliding)"))
                else float(r["score (sliding)"]),
            )
        )

    return rows


def run_graph_kid(csv_path: Path) -> list[KidGraphRow]:
    table = _build_kid_table(csv_path)
    title = "KID Score Comparison"

    print(f"\n[graph] {title}\n")
    print(
        table.to_string(
            float_format=lambda x: "—" if pd.isna(x) else f"{x:.4f}"
        )
    )

    output_path = csv_path.with_suffix(".png")
    _save_table_png(table, output_path, title)

    print(f"\n[ok] graph saved → {output_path}")

    rows: list[KidGraphRow] = []

    for dataset, r in table.iterrows():
        rows.append(
            KidGraphRow(
                dataset=dataset,
                count=int(r["count"]),
                score_inception=None
                if "score (inception)" not in table.columns or pd.isna(r.get("score (inception)"))
                else float(r["score (inception)"]),
                score_dinov2=None
                if "score (dinov2_vitb14)" not in table.columns or pd.isna(r.get("score (dinov2_vitb14)"))
                else float(r["score (dinov2_vitb14)"]),
            )
        )

    return rows


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_graph(path: Path, graph_type: str):
    if graph_type == "CLIP":
        return run_graph_clip(path)
    
    if graph_type == "KID":
        return run_graph_kid(path)

    raise ValueError(f"Unsupported --graph-type: {graph_type}")


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

    return master
