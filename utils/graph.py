"""
-*- coding: utf-8 -*-
Name:        utils/graph.py
Purpose:     Generate tables/graphs from result CSV files.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


class GraphGenerator:
    LINESTYLES = ["-", "--", "-.", ":"]
    MARKERSIZE = 3
    LINEWIDTH = 2

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        self.base = self.df[self.df["transform"] == "none"]
        self.work = self.df[self.df["transform"] != "none"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        self._graph_accuracy_vs_level()
        self._graph_accuracy_vs_lpips()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_figure(self):
        plt.figure()

    def _save(self, suffix: str):
        out = self.csv_path.with_name(f"{self.csv_path.stem}_{suffix}.png")
        plt.savefig(out, dpi=300)
        plt.close()

    def _plot_curve(self, x, y, label, idx):
        plt.plot(
            x,
            y,
            label=label,
            linestyle=self.LINESTYLES[idx % len(self.LINESTYLES)],
            marker="o",
            markersize=self.MARKERSIZE,
            linewidth=self.LINEWIDTH,
        )

    def _with_origin(self, df, x_col, y_col, origin_val):
        return pd.concat([
            pd.DataFrame({x_col: [0 if x_col == "lpips" else 100], y_col: [origin_val]}),
            df[[x_col, y_col]],
        ]).sort_values(x_col)

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------

    def _graph_accuracy_vs_level(self):
        for dataset in self.work["dataset"].unique():
            dsub = self.work[self.work["dataset"] == dataset]
            base_d = self.base[self.base["dataset"] == dataset]

            for transform in dsub["transform"].unique():
                tsub = dsub[dsub["transform"] == transform]

                self._new_figure()

                for i, detector in enumerate(tsub["detector"].unique()):
                    det_sub = tsub[tsub["detector"] == detector].sort_values("level")

                    if not det_sub["accuracy"].notna().any():
                        continue

                    origin_row = base_d[base_d["detector"] == detector]
                    if origin_row.empty:
                        continue

                    origin_val = origin_row["accuracy"].iloc[0]

                    curve = self._with_origin(det_sub, "level", "accuracy", origin_val)

                    self._plot_curve(
                        curve["level"],
                        curve["accuracy"],
                        detector,
                        i,
                    )

                plt.gca().invert_xaxis()
                plt.title(f"{dataset.upper().replace('_', ' ')} – {transform.upper()}\nAccuracy vs Level")
                plt.xlabel("level")
                plt.ylabel("accuracy (%)")
                plt.legend()
                plt.tight_layout()

                self._save(f"acc_vs_level_{dataset}_{transform}")

    def _graph_accuracy_vs_lpips(self):
        directional = {"contrast", "saturation"}

        for dataset in self.work["dataset"].unique():
            dsub = self.work[self.work["dataset"] == dataset]
            base_d = self.base[self.base["dataset"] == dataset]

            for detector in self.df["detector"].unique():
                det_sub = dsub[dsub["detector"] == detector]
                base_det = base_d[base_d["detector"] == detector]

                if base_det.empty or det_sub.empty:
                    continue

                self._new_figure()
                curve_idx = 0
                origin_val = base_det["accuracy"].iloc[0]

                for transform in det_sub["transform"].unique():
                    tsub = det_sub[det_sub["transform"] == transform]

                    if not tsub["accuracy"].notna().any():
                        continue

                    if transform in directional:
                        branches = [
                            ("↓", tsub[tsub["level"] <= 100]),
                            ("↑", tsub[tsub["level"] >= 100]),
                        ]

                        for suffix, branch in branches:
                            branch = branch.sort_values("lpips")
                            if branch.empty:
                                continue

                            curve = self._with_origin(branch, "lpips", "accuracy", origin_val)

                            self._plot_curve(
                                curve["lpips"],
                                curve["accuracy"],
                                f"{transform} {suffix}",
                                curve_idx,
                            )
                            curve_idx += 1
                    else:
                        tsub_sorted = tsub.sort_values("lpips")
                        curve = self._with_origin(tsub_sorted, "lpips", "accuracy", origin_val)

                        self._plot_curve(
                            curve["lpips"],
                            curve["accuracy"],
                            transform,
                            curve_idx,
                        )
                        curve_idx += 1

                plt.title(f"{dataset.upper().replace('_', ' ')} – {detector}\nAccuracy vs LPIPS")
                plt.xlabel("LPIPS")
                plt.ylabel("accuracy (%)")
                plt.legend()

                ax = plt.gca()
                ax.set_xscale("symlog", linthresh=1e-3)
                ax.xaxis.set_major_formatter(plt.ScalarFormatter())
                ax.ticklabel_format(style="plain", axis="x")

                plt.tight_layout()

                self._save(f"acc_vs_lpips_{detector}_{dataset}")


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_graph(csv_path: Path):
    GraphGenerator(csv_path).run()
