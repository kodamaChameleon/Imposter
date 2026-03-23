"""
-*- coding: utf-8 -*-
Name:        utils/graph.py
Purpose:     Generate tables/graphs from result CSV files.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data container (matches style of kid/clip result objects)
# ---------------------------------------------------------------------
LINESTYLES = ["-", "--", "-.", ":"]
MARKERSIZE = 3
LINEWIDTH = 2


# ---------------------------------------------------------------------
# Graphs & Tables
# ---------------------------------------------------------------------


def _graph_test_accuracy_vs_level(df: pd.DataFrame, csv_path: Path):
    base = df[df["transform"] == "none"]
    work = df[df["transform"] != "none"]

    for dataset in work["dataset"].unique():
        dsub = work[work["dataset"] == dataset]
        base_d = base[base["dataset"] == dataset]

        for transform in dsub["transform"].unique():
            tsub = dsub[dsub["transform"] == transform]

            plt.figure()

            for i, detector in enumerate(tsub["detector"].unique()):
                det_sub = tsub[tsub["detector"] == detector].sort_values("level")

                if det_sub["accuracy"].notna().any():
                    origin_row = base_d[base_d["detector"] == detector]

                    if origin_row.empty:
                        continue

                    origin_val = origin_row["accuracy"].iloc[0]

                    curve = pd.concat([
                        pd.DataFrame({"level": [100], "accuracy": [origin_val]}),
                        det_sub[["level", "accuracy"]],
                    ]).sort_values("level")

                    plt.plot(
                        curve["level"],
                        curve["accuracy"],
                        label=detector,
                        linestyle=LINESTYLES[i % len(LINESTYLES)],
                        marker="o",
                        markersize=MARKERSIZE,
                        linewidth=LINEWIDTH,
                    )

            plt.gca().invert_xaxis()
            plt.title(f"{dataset.upper().replace('_', ' ')} – {transform.upper()}\nAccuracy vs Level")
            plt.xlabel("level")
            plt.ylabel("accuracy (%)")
            plt.legend()
            plt.tight_layout()

            out = csv_path.with_name(
                f"{csv_path.stem}_acc_vs_level_{dataset}_{transform}.png"
            )
            plt.savefig(out, dpi=300)
            plt.close()


def _graph_test_accuracy_vs_lpips(df: pd.DataFrame, csv_path: Path):
    detectors = df["detector"].unique()

    base = df[df["transform"] == "none"]
    work = df[df["transform"] != "none"]

    directional_transforms = {"contrast", "saturation"}

    for dataset in work["dataset"].unique():
        dsub = work[work["dataset"] == dataset]
        base_d = base[base["dataset"] == dataset]

        for det_name in detectors:
            plt.figure()
            curve_index = 0

            det_sub = dsub[dsub["detector"] == det_name]
            base_det = base_d[base_d["detector"] == det_name]

            if base_det.empty or det_sub.empty:
                continue

            origin_val = base_det["accuracy"].iloc[0]

            for transform in det_sub["transform"].unique():
                tsub = det_sub[det_sub["transform"] == transform]

                if not tsub["accuracy"].notna().any():
                    continue

                # ----------------------------------------
                # Directional transforms
                # ----------------------------------------
                if transform in directional_transforms:

                    # Decrease branch
                    dec = (
                        tsub[tsub["level"] <= 100]
                        .sort_values("lpips")
                    )

                    if not dec.empty:
                        curve = pd.concat(
                            [
                                pd.DataFrame({"lpips": [0], "accuracy": [origin_val]}),
                                dec[["lpips", "accuracy"]],
                            ]
                        ).sort_values("lpips")

                        plt.plot(
                            curve["lpips"],
                            curve["accuracy"],
                            label=f"{transform} ↓",
                            linestyle=LINESTYLES[curve_index % len(LINESTYLES)],
                            marker="o",
                            markersize=MARKERSIZE,
                            linewidth=LINEWIDTH,
                        )
                        curve_index += 1

                    # Increase branch
                    inc = (
                        tsub[tsub["level"] >= 100]
                        .sort_values("lpips")
                    )

                    if not inc.empty:
                        curve = pd.concat(
                            [
                                pd.DataFrame({"lpips": [0], "accuracy": [origin_val]}),
                                inc[["lpips", "accuracy"]],
                            ]
                        ).sort_values("lpips")

                        plt.plot(
                            curve["lpips"],
                            curve["accuracy"],
                            label=f"{transform} ↑",
                            linestyle=LINESTYLES[curve_index % len(LINESTYLES)],
                            marker="o",
                            markersize=MARKERSIZE,
                            linewidth=LINEWIDTH,
                        )
                        curve_index += 1

                # ----------------------------------------
                # Normal transforms
                # ----------------------------------------
                else:
                    tsub_sorted = tsub.sort_values("lpips")

                    curve = pd.concat(
                        [
                            pd.DataFrame({"lpips": [0], "accuracy": [origin_val]}),
                            tsub_sorted[["lpips", "accuracy"]],
                        ]
                    ).sort_values("lpips")

                    plt.plot(
                        curve["lpips"],
                        curve["accuracy"],
                        label=transform,
                        linestyle=LINESTYLES[curve_index % len(LINESTYLES)],
                        marker="o",
                        markersize=MARKERSIZE,
                        linewidth=LINEWIDTH,
                    )
                    curve_index += 1

            plt.title(f"{dataset.upper().replace('_', ' ')} – {det_name}\nAccuracy vs LPIPS")
            plt.xlabel("LPIPS")
            plt.ylabel("accuracy (%)")
            plt.legend()

            ax = plt.gca()
            ax.set_xscale("symlog", linthresh=1e-3)
            ax.xaxis.set_major_formatter(plt.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="x")

            plt.tight_layout()

            out = csv_path.with_name(
                f"{csv_path.stem}_acc_vs_lpips_{det_name}_{dataset}.png"
            )
            plt.savefig(out, dpi=300)
            plt.close()


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_graph(csv_path: Path):
    df = pd.read_csv(csv_path)

    _graph_test_accuracy_vs_level(df, csv_path)
    _graph_test_accuracy_vs_lpips(df, csv_path)

    print("[ok] test graphs complete")
