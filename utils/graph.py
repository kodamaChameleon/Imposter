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
LINESTYLES = ["-", "--", "-.", ":"]
MARKERSIZE = 3
LINEWIDTH = 2

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
# Graphs & Tables
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


def _table_test_baseline(df: pd.DataFrame, csv_path: Path):
    title = "Baseline Test Results"

    base = (
        df[df["transform"] == "none"]
        .sort_values("dataset")
        .set_index("dataset")[
            [
                "UFD_accuracy",
                "UFD_precision",
                "SAFE_accuracy",
                "SAFE_precision",
            ]
        ]
        .rename(
            columns={
                "UFD_accuracy": "UFD_acc",
                "UFD_precision": "UFD_ap",
                "SAFE_accuracy": "SAFE_acc",
                "SAFE_precision": "SAFE_ap",
            }
        )
    )

    print(f"\n[graph] {title}\n")
    print(base.to_string(float_format=lambda x: f"{x:.4f}"))

    out = csv_path.with_name(csv_path.stem + "_baseline.png")
    _save_table_png(base, out, title)


def _graph_test_accuracy_vs_level(df: pd.DataFrame, csv_path: Path):
    detectors = {
        "UFD": "UFD_accuracy",
        "SAFE": "SAFE_accuracy",
    }

    base = df[df["transform"] == "none"].set_index("dataset")
    work = df[df["transform"] != "none"]

    for dataset in work["dataset"].unique():
        dsub = work[work["dataset"] == dataset]

        for transform in dsub["transform"].unique():
            tsub = dsub[dsub["transform"] == transform].sort_values("level")

            plt.figure()

            for i, (det_name, col) in enumerate(detectors.items()):
                if tsub[col].notna().any():
                    origin_val = base.loc[dataset, col]

                    curve = pd.concat(
                        [
                            pd.DataFrame({"level": [100], col: [origin_val]}),
                            tsub[["level", col]],
                        ]
                    ).sort_values("level")

                    plt.plot(
                        curve["level"],
                        curve[col],
                        label=det_name,
                        linestyle=LINESTYLES[i % len(LINESTYLES)],
                        marker="o",
                        markersize=MARKERSIZE,
                        linewidth=LINEWIDTH,
                    )

            plt.gca().invert_xaxis()
            plt.title(f"{dataset.upper().replace('_', ' ')} – {transform.upper()}\nAccuracy vs Level")
            plt.xlabel("level")
            plt.ylabel("accuracy")
            plt.legend()
            plt.tight_layout()

            out = csv_path.with_name(
                f"acc_vs_level_{dataset}_{transform}.png"
            )
            plt.savefig(out, dpi=300)
            plt.close()


def _graph_test_accuracy_vs_lpips(df: pd.DataFrame, csv_path: Path):
    detectors = {
        "UFD": "UFD_accuracy",
        "SAFE": "SAFE_accuracy",
    }

    base = df[df["transform"] == "none"].set_index("dataset")
    work = df[df["transform"] != "none"]

    for dataset in work["dataset"].unique():
        dsub = work[work["dataset"] == dataset]

        for det_name, col in detectors.items():
            plt.figure()

            for i, transform in enumerate(dsub["transform"].unique()):
                tsub = (
                    dsub[dsub["transform"] == transform]
                    .sort_values("lpips")
                )

                if tsub[col].notna().any():
                    origin_val = base.loc[dataset, col]

                    curve = pd.concat(
                        [
                            pd.DataFrame({"lpips": [0], col: [origin_val]}),
                            tsub[["lpips", col]],
                        ]
                    ).sort_values("lpips")

                    plt.plot(
                        curve["lpips"],
                        curve[col],
                        label=transform,
                        linestyle=LINESTYLES[i % len(LINESTYLES)],
                        marker="o",
                        markersize=MARKERSIZE,
                        linewidth=LINEWIDTH,
                    )

            plt.title(f"{dataset.upper().replace('_', ' ')} – {det_name}\nAccuracy vs LPIPS")
            plt.xlabel("LPIPS")
            plt.ylabel("accuracy")
            plt.legend()

            # Adjust X scale
            ax = plt.gca()
            ax.set_xscale("symlog", linthresh=1e-3)
            ax.xaxis.set_major_formatter(plt.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="x")

            plt.tight_layout()

            out = csv_path.with_name(
                f"acc_vs_lpips_{det_name}_{dataset}.png"
            )
            plt.savefig(out, dpi=300)
            plt.close()


def _save_table_png(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    *,
    float_precision: int = 4,
    na_rep: str = "—",
) -> None:
    display_df = df.copy()

    # -----------------------------
    # dtype-driven formatting
    # -----------------------------
    for col in display_df.columns:
        series = display_df[col]

        if pd.api.types.is_integer_dtype(series):
            display_df[col] = series.map(
                lambda x: na_rep if pd.isna(x) else f"{int(x):,}"
            )

        elif pd.api.types.is_float_dtype(series):
            fmt = f"{{:.{float_precision}f}}"
            display_df[col] = series.map(
                lambda x: na_rep if pd.isna(x) else fmt.format(x)
            )

        else:
            display_df[col] = series.fillna(na_rep)

    # -----------------------------
    # dynamic sizing
    # -----------------------------
    n_rows, n_cols = display_df.shape

    fig_width = 7.5
    fig_height = n_rows * 0.4

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
    table.set_fontsize(10)
    table.scale(1, 1.35)

    # -----------------------------
    # header styling
    # -----------------------------
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAF2")

    # -----------------------------
    # zebra striping
    # -----------------------------
    for r in range(1, n_rows + 1):
        if r % 2 == 0:
            for c in range(n_cols):
                table[(r, c)].set_facecolor("#F7F7F7")

    ax.set_title(title, fontsize=14, weight="bold", pad=4)

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


def run_graph_test(csv_path: Path):
    df = pd.read_csv(csv_path)

    _table_test_baseline(df, csv_path)
    _graph_test_accuracy_vs_level(df, csv_path)
    _graph_test_accuracy_vs_lpips(df, csv_path)

    print("[ok] test graphs complete")


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_graph(path: Path, graph_type: str):
    if graph_type == "CLIP":
        return run_graph_clip(path)
    
    if graph_type == "KID":
        return run_graph_kid(path)

    if graph_type == "TEST":
        return run_graph_test(path)

    raise ValueError(f"Unsupported --graph-type: {graph_type}")
