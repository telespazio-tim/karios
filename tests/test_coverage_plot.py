#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the key point coverage (density/quality) plot."""

from unittest.mock import Mock

import numpy as np
import pytest
from pandas import DataFrame

from karios.core.configuration import CoveragePlotConfiguration
from karios.core.image import GdalRasterImage
from karios.report.coverage_plot import (
    CoveragePlot,
    compute_density_grid,
    compute_quality_grid,
    coverage_statistics,
    quality_scalar_stats,
)


def test_compute_density_grid_counts_points_per_cell():
    """One point per corner of a 2x2 grid must land in its own cell."""
    points = DataFrame({"x0": [1, 99, 1, 99], "y0": [1, 1, 99, 99]})

    counts = compute_density_grid(points, x_size=100, y_size=100, grid_size=2)

    assert counts.shape == (2, 2)
    assert counts.sum() == 4
    assert np.array_equal(counts, np.array([[1, 1], [1, 1]]))


def test_compute_quality_grid_averages_only_points_in_cell():
    """A cell's mean must average only the points that landed in it, and
    empty cells must be NaN rather than 0 (0 would look like a bad match,
    not an absence of one)."""
    points = DataFrame({"x0": [1, 2, 99], "y0": [1, 1, 1], "score": [0.2, 0.8, 0.5]})

    mean_score = compute_quality_grid(points, "score", x_size=100, y_size=100, grid_size=2)

    assert mean_score[0, 0] == pytest.approx(0.5)  # (0.2 + 0.8) / 2
    assert mean_score[0, 1] == pytest.approx(0.5)
    assert np.isnan(mean_score[1, 0])
    assert np.isnan(mean_score[1, 1])


def test_compute_quality_grid_ignores_nan_points():
    """A NaN quality value (e.g. ZNCC/NMI not computed for that KP) must be
    excluded from its cell's mean, not poison the whole cell to NaN."""
    points = DataFrame(
        {"x0": [1, 2], "y0": [1, 1], "zncc_score": [0.6, np.nan]},
    )

    mean_zncc = compute_quality_grid(points, "zncc_score", x_size=100, y_size=100, grid_size=2)

    assert mean_zncc[0, 0] == pytest.approx(0.6)


def test_compute_quality_grid_missing_column_is_all_nan():
    """A column entirely absent (e.g. large shift detection applied) must
    yield an all-NaN grid rather than raising."""
    points = DataFrame({"x0": [1, 2], "y0": [1, 1]})

    grid = compute_quality_grid(points, "zncc_score", x_size=100, y_size=100, grid_size=2)

    assert grid.shape == (2, 2)
    assert np.all(np.isnan(grid))


def test_coverage_statistics_uniform_is_more_uniform_than_clustered():
    """A perfectly even grid must score a lower coefficient of variation
    (more uniform) than one where all points pile into a single cell."""
    uniform = np.full((4, 4), 10)
    clustered = np.zeros((4, 4))
    clustered[0, 0] = 160

    uniform_stats = coverage_statistics(uniform)
    clustered_stats = coverage_statistics(clustered)

    assert uniform_stats["total_points"] == clustered_stats["total_points"] == 160
    assert uniform_stats["coefficient_of_variation"] < clustered_stats["coefficient_of_variation"]
    assert uniform_stats["empty_cells"] == 0
    assert clustered_stats["empty_cells"] == 15


def test_coverage_statistics_handles_no_points():
    """An all-empty grid must not raise a division error."""
    stats = coverage_statistics(np.zeros((3, 3)))

    assert stats["total_points"] == 0
    assert stats["empty_cells"] == 9
    assert np.isnan(stats["coefficient_of_variation"])


def test_quality_scalar_stats_mean_and_threshold_ratio():
    """Mean and above-threshold ratio must be computed over valid (non-NaN)
    points only."""
    points = DataFrame({"zncc_score": [0.9, 0.7, np.nan, 0.5]})

    stats = quality_scalar_stats(points, "zncc_score", threshold=0.8)

    assert stats["computed"] is True
    assert stats["valid_count"] == 3
    assert stats["mean"] == pytest.approx((0.9 + 0.7 + 0.5) / 3)
    assert stats["above_threshold_ratio"] == pytest.approx(1 / 3)  # only 0.9 >= 0.8


def test_quality_scalar_stats_missing_column():
    """A column entirely absent must report computed=False rather than raising."""
    stats = quality_scalar_stats(DataFrame({"x0": [1, 2]}), "zncc_score")

    assert stats["computed"] is False
    assert stats["valid_count"] == 0
    assert np.isnan(stats["mean"])


def test_quality_scalar_stats_no_valid_points():
    """A column present but entirely NaN must give NaN mean/ratio, not raise."""
    points = DataFrame({"zncc_score": [np.nan, np.nan]})

    stats = quality_scalar_stats(points, "zncc_score", threshold=0.8)

    assert stats["computed"] is True
    assert stats["valid_count"] == 0
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["above_threshold_ratio"])


def _mon_image(size=1000):
    mon_image = Mock(spec=GdalRasterImage)
    mon_image.x_size = size
    mon_image.y_size = size
    return mon_image


def test_coverage_plot_smoke(tmp_path):
    """The plot must render and save without crashing on real-shaped data,
    including the ZNCC/NMI panels."""
    rng = np.random.default_rng(0)
    points = DataFrame(
        {
            "x0": rng.uniform(0, 1000, 200),
            "y0": rng.uniform(0, 1000, 200),
            "score": rng.uniform(0, 1, 200),
            "zncc_score": rng.uniform(-1, 1, 200),
            "mi_score": rng.uniform(0, 1, 200),
        }
    )

    plot = CoveragePlot(CoveragePlotConfiguration(), _mon_image(), points, prefix=None)
    output_file = tmp_path / "coverage.png"
    plot.plot(output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_coverage_plot_smoke_without_zncc_nmi(tmp_path):
    """Must still render when zncc_score/mi_score are absent entirely (large
    shift detection applied), showing a placeholder instead of crashing."""
    rng = np.random.default_rng(0)
    points = DataFrame(
        {
            "x0": rng.uniform(0, 1000, 50),
            "y0": rng.uniform(0, 1000, 50),
            "score": rng.uniform(0, 1, 50),
        }
    )

    plot = CoveragePlot(CoveragePlotConfiguration(), _mon_image(), points, prefix=None)
    output_file = tmp_path / "coverage.png"
    plot.plot(output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0
