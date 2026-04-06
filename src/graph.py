"""
-*- coding: utf-8 -*-
Name:        utils/graph.py
Purpose:     Generate tables/graphs from result CSV files.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        self._graph_confusion_matrices(normalize=True)

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
        out = self.csv_path.with_name(f"{self.csv_path.stem}_{suffix.replace(' ', '_')}.png")
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
        for dataset in tqdm(
            self.work["dataset"].unique(),
            desc=f"{y_col} datasets",
            leave=False
        ):
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
                    f"{detector} – {dataset.replace('_', ' ')}\n"
                    f"{y_col.replace('_', ' ')} vs LPIPS".upper()
                )
                plt.xlabel("lpips")
                plt.ylabel(y_col.replace("_", " "))
                plt.legend()

                ax = plt.gca()
                ax.set_xscale("symlog", linthresh=1e-3)
                ax.xaxis.set_major_formatter(plt.ScalarFormatter())
                ax.ticklabel_format(style="plain", axis="x")

                plt.tight_layout()

                self._save(f"{y_col}_vs_lpips_{detector}_{dataset}")

    def _graph_confusion_matrices(self, normalize: bool = False, max_cols: int = 7):

        origin = self.level_origin

        for detector in self.df["detector"].unique():
            for dataset in self.df["dataset"].unique():

                subset_df = self.df[
                    (self.df["detector"] == detector) &
                    (self.df["dataset"] == dataset)
                ]

                if subset_df.empty:
                    continue

                # -----------------------------
                # Build rows: (transform, direction)
                # -----------------------------
                rows = []

                for transform in subset_df["transform"].unique():
                    if transform == "none":
                        continue

                    tsub = subset_df[subset_df["transform"] == transform]

                    has_lower = (tsub["level"] < origin).any()
                    has_upper = (tsub["level"] > origin).any()

                    if has_lower:
                        rows.append((transform, "↓"))

                    if has_upper:
                        rows.append((transform, "↑"))

                if not rows:
                    continue

                # Optional: sort rows nicely
                rows = sorted(rows, key=lambda x: (x[0], x[1] == "↑"))

                # -----------------------------
                # Build columns: Δ levels
                # -----------------------------
                deltas = sorted({
                    abs(level - origin)
                    for level in subset_df["level"].unique()
                })

                # Always keep baseline (0), then smallest deltas
                deltas = [0] + sorted([d for d in deltas if d != 0])[: max_cols - 1]

                n_rows = len(rows)
                n_cols = len(deltas)

                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(1.5 * n_cols, 1.5 * n_rows),
                    squeeze=False
                )

                # -----------------------------
                # Global color scale
                # -----------------------------
                if normalize:
                    vmin, vmax = 0, 1
                else:
                    vmax = subset_df[["tn", "fp", "fn", "tp"]].values.max()
                    vmin = 0

                last_row = n_rows - 1
                for i, (transform, direction) in enumerate(rows):
                    for j, delta in enumerate(deltas):
                        ax = axes[i][j]

                        # -----------------------------
                        # Select correct cell
                        # -----------------------------
                        if delta == 0:
                            cell = subset_df[
                                subset_df["transform"] == "none"
                            ]
                        else:
                            level = origin - delta if direction == "↓" else origin + delta

                            cell = subset_df[
                                (subset_df["transform"] == transform) &
                                (subset_df["level"] == level)
                            ]

                        if cell.empty:
                            ax.axis("off")
                            continue

                        row = cell.iloc[0]

                        tn, fp, fn, tp = row["tn"], row["fp"], row["fn"], row["tp"]

                        raw_cm = np.array([[tn, fp], [fn, tp]], dtype=float)
                        cm = raw_cm.copy()

                        if normalize:
                            row_sums = cm.sum(axis=1, keepdims=True)
                            row_sums[row_sums == 0] = 1
                            cm = cm / row_sums

                        cmap = "Oranges_r"
                        im = ax.imshow(cm, vmin=vmin, vmax=vmax, cmap=cmap)

                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        
                        # -----------------------------
                        # Column titles (Δ space)
                        # -----------------------------
                        if i == 0:
                            if delta == 0:
                                label = "baseline"
                            else:
                                label = f"Δ{int(delta)}"
                            ax.set_title(label)

                        # -----------------------------
                        # Row labels
                        # -----------------------------
                        if j == 0 and i != last_row:
                            ax.set_ylabel(f"{transform} {direction}", labelpad=16)
                        
                        elif i == last_row and j == 0:
                            ax.set_ylabel(f"{transform} {direction}", labelpad=4)

                            ax.set_xticks([0, 1])
                            ax.set_yticks([0, 1])
                            ax.set_xticklabels(["0", "1"])
                            ax.set_yticklabels(["0", "1"])    

                        # -----------------------------
                        # Annotate values
                        # -----------------------------
                        for x in range(2):
                            for y in range(2):
                                raw_val = int(raw_cm[x, y])

                                if normalize:
                                    norm_val = cm[x, y]
                                    text = f"{norm_val:.2f}\n({raw_val})"
                                else:
                                    text = f"{raw_val}"

                                ax.text(
                                    y,
                                    x,
                                    text,
                                    ha="center",
                                    va="center",
                                    fontsize=8
                                )

                        # -----------------------------
                        # Highlight baseline column
                        # -----------------------------
                        if delta == 0:
                            ax.set_facecolor("#f5f5f5")

                # -----------------------------
                # Title + colorbar
                # -----------------------------
                if dataset.lower() == 'average':
                    fig.suptitle(
                        f"{detector} – All Datasets".upper().replace('_', ' '),
                        fontsize=16
                    )
                else:
                    fig.suptitle(
                        f"{detector} – {dataset}".upper().replace('_', ' '),
                        fontsize=16
                    )


                plt.tight_layout(rect=[0.03,0.01,0.95,1])
                plt.subplots_adjust(wspace=0.06, hspace=0.02)
                fig.supxlabel("Predicted")
                fig.supylabel("True")
                cbar_ax = fig.add_axes([0.945, 0.1, 0.02, 0.8])
                fig.colorbar(im, cax=cbar_ax)
                

                self._save(f"confusion_{detector}_{dataset}")


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_graph(csv_path: Path, metrics: list):
    GraphGenerator(csv_path).run(metrics=metrics)
