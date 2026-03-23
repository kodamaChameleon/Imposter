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

    def __init__(self, csv_path: Path, level_origin: float = 100):
        self.csv_path = csv_path
        self.level_origin = level_origin

        self.df = pd.read_csv(csv_path)

        self.base = self.df[self.df["transform"] == "none"]
        self.work = self.df[self.df["transform"] != "none"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, metrics: list):
        for metric in metrics:
            try:
                self._graph_metric_vs_lpips(metric)
                print(f"[ok] {metric} graphs complete")
            except ValueError as e:
                print(f"[error] {e}")
                continue

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

    def _with_origin(self, df, y_col, origin_val):
        return pd.concat([
            pd.DataFrame({"lpips": [0], y_col: [origin_val]}),
            df[["lpips", y_col]],
        ]).sort_values("lpips")

    def _split_directional(self, tsub: pd.DataFrame):
        has_lower = (tsub["level"] < self.level_origin).any()
        has_upper = (tsub["level"] > self.level_origin).any()

        if has_lower and has_upper:
            return [
                ("↓", tsub[tsub["level"] < self.level_origin]),
                ("↑", tsub[tsub["level"] > self.level_origin]),
            ]

        return [(None, tsub)]

    # ------------------------------------------------------------------
    # Core graph
    # ------------------------------------------------------------------

    def _graph_metric_vs_lpips(self, y_col: str):
        for dataset in self.work["dataset"].unique():
            dsub = self.work[self.work["dataset"] == dataset]
            base_d = self.base[self.base["dataset"] == dataset]

            for detector in self.df["detector"].unique():
                det_sub = dsub[dsub["detector"] == detector]
                base_det = base_d[base_d["detector"] == detector]

                if base_det.empty or det_sub.empty:
                    continue

                if y_col not in det_sub.columns:
                    raise ValueError(f"Metric '{y_col}' not found in CSV columns")

                self._new_figure()
                curve_idx = 0

                origin_val = base_det[y_col].iloc[0]

                for transform in det_sub["transform"].unique():
                    tsub = det_sub[det_sub["transform"] == transform]

                    if not tsub[y_col].notna().any():
                        continue

                    splits = self._split_directional(tsub)

                    for suffix, subset in splits:
                        subset = subset.sort_values("lpips")
                        if subset.empty:
                            continue

                        curve = self._with_origin(subset, y_col, origin_val)

                        label = transform if suffix is None else f"{transform} {suffix}"

                        self._plot_curve(
                            curve["lpips"],
                            curve[y_col],
                            label,
                            curve_idx,
                        )
                        curve_idx += 1

                plt.title(
                    f"{detector} – {dataset.upper().replace('_', ' ')}\n"
                    f"{y_col.replace('_', ' ').title()} vs LPIPS"
                )
                plt.xlabel("LPIPS")
                plt.ylabel(y_col.replace("_", " "))
                plt.legend()

                ax = plt.gca()
                ax.set_xscale("symlog", linthresh=1e-3)
                ax.xaxis.set_major_formatter(plt.ScalarFormatter())
                ax.ticklabel_format(style="plain", axis="x")

                plt.tight_layout()

                self._save(f"{y_col}_vs_lpips_{detector}_{dataset}")


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_graph(csv_path: Path, metrics: list):
    GraphGenerator(csv_path).run(metrics=metrics)
