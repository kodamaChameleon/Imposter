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
# CLIP graph
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