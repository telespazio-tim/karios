# -*- coding: utf-8 -*-
# Copyright (c) 2026 Telespazio France.
#
# This file is part of KARIOS.
# See https://github.com/telespazio-tim/karios for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module to plot key point spatial coverage and local match quality.

A raw key point count says nothing about where those points are: they can
cluster in the most textured regions of a scene and leave the rest
unmatched, which the accuracy statistics (computed over all points at once)
cannot reveal. This module grids the monitored image and reports, per cell,
how many key points landed there and how good those matches were on
average - density next to three independent quality signals (KLT score,
ZNCC, NMI), since they can disagree: a KP the tracker itself is confident
about can still be a poor radiometric match - plus grid-level summary
statistics (empty cell ratio, coefficient of variation of the per-cell
count) that quantify how uniform the coverage actually is.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame

from karios.core.configuration import CoveragePlotConfiguration
from karios.core.image import GdalRasterImage
from karios.report.commons import AbstractPlot

logger = logging.getLogger(__name__)


def _grid_edges(x_size: int, y_size: int, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(0, x_size, grid_size + 1), np.linspace(0, y_size, grid_size + 1)


def compute_density_grid(points: DataFrame, x_size: int, y_size: int, grid_size: int) -> np.ndarray:
    """Count key points per cell of a `grid_size` x `grid_size` grid over the image extent.

    Args:
        points: KP data frame with series x0, y0
        x_size: monitored image width in pixels
        y_size: monitored image height in pixels
        grid_size: number of bins along each axis

    Returns:
        NDArray: counts, shaped (grid_size, grid_size), indexed [row=y, col=x].
    """
    x_edges, y_edges = _grid_edges(x_size, y_size, grid_size)
    counts, _, _ = np.histogram2d(
        points["x0"].to_numpy(), points["y0"].to_numpy(), bins=[x_edges, y_edges]
    )
    return counts.T  # histogram2d is [x, y]; grid convention here is [row=y, col=x]


def compute_quality_grid(
    points: DataFrame, column: str, x_size: int, y_size: int, grid_size: int
) -> np.ndarray:
    """Average a KP quality column per cell of a `grid_size` x `grid_size` grid.

    NaN-safe: points where `column` is NaN (not computed for that KP - e.g.
    ZNCC/NMI are only evaluated above the confidence threshold) are excluded
    from both the sum and the count, rather than poisoning their cell's mean.

    Args:
        points: KP data frame with series x0, y0, and (optionally) `column`
        column: name of the quality column to average, e.g. "score",
            "zncc_score", "mi_score"
        x_size: monitored image width in pixels
        y_size: monitored image height in pixels
        grid_size: number of bins along each axis

    Returns:
        NDArray: mean value per cell, shaped (grid_size, grid_size), indexed
            [row=y, col=x]. NaN where the column is absent entirely (e.g.
            large shift detection was applied) or no KP in that cell has a
            valid value.
    """
    x_edges, y_edges = _grid_edges(x_size, y_size, grid_size)
    grid_shape = (grid_size, grid_size)

    if column not in points:
        return np.full(grid_shape, np.nan)

    valid = points[points[column].notna()]
    if valid.empty:
        return np.full(grid_shape, np.nan)

    x = valid["x0"].to_numpy()
    y = valid["y0"].to_numpy()
    sum_value, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=valid[column])
    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_value = np.where(count > 0, sum_value / count, np.nan)

    return mean_value.T


def quality_scalar_stats(
    points: DataFrame, column: str, threshold: float | None = None
) -> dict[str, float | int | bool]:
    """Overall (non-spatial) summary of a KP quality column.

    Complements `compute_quality_grid`'s per-cell spatial means with the
    single scene-wide number: how good matching is *on average*, and - when
    a threshold is given - what fraction of KP actually clear it. ZNCC in
    particular is conventionally trusted only above ~0.8; that is a
    statement about individual KP, not about a cell average, so it is
    checked against the raw per-point values here rather than the grid.

    Args:
        points: KP data frame, optionally containing `column`
        column: name of the quality column, e.g. "zncc_score", "mi_score"
        threshold: if given, also report the fraction of valid KP at or
            above it

    Returns:
        dict: `computed` (whether the column exists at all), `valid_count`
            (non-NaN KP), `mean` (NaN if no valid KP), and - only when
            `threshold` is given - `above_threshold_ratio` (NaN if no valid
            KP).
    """
    if column not in points:
        return {"computed": False, "valid_count": 0, "mean": float("nan")}

    valid = points[column].dropna()
    stats: dict[str, float | int | bool] = {
        "computed": True,
        "valid_count": int(len(valid)),
        "mean": float(valid.mean()) if len(valid) else float("nan"),
    }
    if threshold is not None:
        stats["above_threshold_ratio"] = (
            float((valid >= threshold).mean()) if len(valid) else float("nan")
        )
    return stats


def coverage_statistics(counts: np.ndarray) -> dict[str, float]:
    """Summarize how uniformly key points cover the grid.

    Args:
        counts: per-cell key point count, as returned by `compute_density_grid`

    Returns:
        dict[str, float]: total_points, empty_cells, total_cells,
            empty_cell_ratio, and coefficient_of_variation (std / mean of the
            per-cell count, NaN when there are no points at all - lower means
            more uniform coverage).
    """
    total_cells = counts.size
    empty_cells = int(np.count_nonzero(counts == 0))
    mean = counts.mean()
    cv = float(counts.std() / mean) if mean > 0 else float("nan")

    return {
        "total_points": int(counts.sum()),
        "empty_cells": empty_cells,
        "total_cells": total_cells,
        "empty_cell_ratio": empty_cells / total_cells,
        "coefficient_of_variation": cv,
    }


# (title, column, colorbar label, vmin, vmax, config attr for a pass/fail
# threshold on this column, or None). ZNCC is conventionally trusted only
# above ~0.8; KLT score and NMI have no such established bar here.
_QUALITY_PANELS = [
    ("Mean KLT score", "score", "mean KLT score", 0.0, 1.0, None),
    ("Mean ZNCC", "zncc_score", "mean ZNCC", -1.0, 1.0, "zncc_threshold"),
    ("Mean NMI", "mi_score", "mean NMI", 0.0, 1.0, None),
]


class CoveragePlot(AbstractPlot):
    # pylint: disable=too-few-public-methods
    """Coverage plot class. It plots:
    - key point density: count of key points per grid cell
    - three key point quality signals per grid cell: mean KLT score, mean
      ZNCC, mean NMI (the latter two absent when large shift detection was
      applied, since they are not computed in that case)
    and annotates the density panel with grid-level coverage statistics.
    """

    def __init__(
        self,
        config: CoveragePlotConfiguration,
        mon_image: GdalRasterImage,
        points: DataFrame,
        prefix: str | None,
    ):
        """Constructor

        Args:
            config (CoveragePlotConfiguration): plot config
            mon_image (GdalRasterImage): monitored image, for its extent
            points (DataFrame): KP data frame with series x0, y0, score, and
                optionally zncc_score, mi_score
            prefix (str|None): figure title prefix
        """
        super().__init__(prefix, config.fig_size)
        self._config = config
        self._mon_img = mon_image
        self._points = points

    ####################################################
    # Abstract implementation
    #

    @property
    def _figure_title(self) -> str:
        return "Key point coverage: density and quality"

    def _prepare_figure(self, fig_size) -> Figure:
        # A finer grid needs more pixels per cell to stay legible, so this
        # plot renders at a higher DPI than the other report figures rather
        # than sharing AbstractPlot's default.
        return plt.figure(figsize=(fig_size * 1.5, fig_size * 1.5), dpi=150)

    def _plot(self):
        grid_size = self._config.grid_size
        x_size, y_size = self._mon_img.x_size, self._mon_img.y_size
        extent = [0, x_size, y_size, 0]
        x_edges, y_edges = _grid_edges(x_size, y_size, grid_size)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        counts = compute_density_grid(self._points, x_size, y_size, grid_size)

        density_ax = self._figure.add_subplot(2, 2, 1)
        self._plot_density(density_ax, counts, extent)

        for position, (title, column, cbar_label, vmin, vmax, threshold_attr) in enumerate(
            _QUALITY_PANELS, start=2
        ):
            axes = self._figure.add_subplot(2, 2, position)
            if column not in self._points:
                self._plot_not_computed(axes, extent, title)
                continue
            threshold = getattr(self._config, threshold_attr) if threshold_attr else None
            grid = compute_quality_grid(self._points, column, x_size, y_size, grid_size)
            self._plot_quality(
                axes,
                grid,
                extent,
                x_centers,
                y_centers,
                title,
                column,
                cbar_label,
                vmin,
                vmax,
                threshold,
            )

        # Leave room at the top for the suptitle AbstractPlot.plot() adds
        # after _plot() returns, and at the bottom for the stats text.
        self._figure.tight_layout(rect=(0, 0.04, 1, 0.94))

    ####################################################
    # Helper methods
    #

    def _plot_density(self, axes, counts: np.ndarray, extent: list) -> None:
        axes.set_title("Key point density")
        image = axes.imshow(counts, extent=extent, cmap="viridis", aspect="auto")
        self._figure.colorbar(image, ax=axes, label="key points / cell")

        stats = coverage_statistics(counts)
        stats_text = (
            f"Total KP: {stats['total_points']}   "
            f"Empty cells: {stats['empty_cells']}/{stats['total_cells']} "
            f"({100 * stats['empty_cell_ratio']:.1f}%)   "
            f"Coverage CV: {stats['coefficient_of_variation']:.2f} (lower = more uniform)"
        )
        self._figure.text(0.5, 0.01, stats_text, ha="center", va="bottom", size="10")

    def _plot_not_computed(self, axes, extent: list, title: str) -> None:
        """Placeholder for a quality column that was never computed at all
        (as opposed to computed but NaN for a given cell/point)."""
        axes.set_title(title)
        axes.set_xlim(extent[0], extent[1])
        axes.set_ylim(extent[2], extent[3])
        axes.text(
            0.5,
            0.5,
            "not computed\n(large shift detection applied)",
            transform=axes.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

    def _plot_quality(
        self,
        axes,
        grid: np.ndarray,
        extent: list,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        title: str,
        column: str,
        cbar_label: str,
        vmin: float,
        vmax: float,
        threshold: float | None,
    ) -> None:
        cmap = plt.get_cmap(self._config.quality_colormap).copy()
        cmap.set_bad(color="lightgray")
        image = axes.imshow(
            np.ma.masked_invalid(grid),
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        self._figure.colorbar(image, ax=axes, label=cbar_label)

        stats = quality_scalar_stats(self._points, column, threshold)
        caption = f"mean={stats['mean']:.2f}"
        if threshold is not None and not np.isnan(stats.get("above_threshold_ratio", np.nan)):
            caption += f"   >= {threshold:.2f}: {100 * stats['above_threshold_ratio']:.1f}% of KP"
        axes.set_title(f"{title}\n{caption}", fontsize=10)

        # Draw the threshold as an explicit contour on the spatial map, not
        # just a number: this is what actually shows *where* matching falls
        # short of it, not merely how often. Guard against an all-NaN grid
        # (e.g. every KP below the confidence threshold) to avoid a spurious
        # "All-NaN slice" warning from nanmin/nanmax.
        finite = grid[np.isfinite(grid)]
        if threshold is not None and finite.size and finite.min() < threshold < finite.max():
            axes.contour(
                x_centers,
                y_centers,
                grid,
                levels=[threshold],
                colors="black",
                linewidths=1.2,
                linestyles="dashed",
            )
