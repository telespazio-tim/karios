#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the overview plot."""

from unittest.mock import Mock

import numpy as np
from pandas import DataFrame

from karios.core.configuration import OverviewPlotConfiguration
from karios.core.image import GdalRasterImage
from karios.report.overview_plot import OverviewPlot


def _mock_image(size=64):
    img = Mock(spec=GdalRasterImage)
    img.array = np.ones((size, size), dtype=np.uint8) * 128
    img.file_name = "image.tif"
    img.filepath = "/fake/image.tif"
    img.no_data_value = None
    img.x_size = size
    img.y_size = size
    return img


def test_radial_error_plot_survives_a_perfect_match(tmp_path):
    """A perfect match (e.g. an image matched against itself) has zero
    radial error everywhere. With shift_auto_axes_limit, the color scale's
    vmin/vcenter/vmax would then all collapse to 0, which matplotlib's
    TwoSlopeNorm rejects as not strictly ascending - this must be guarded
    against rather than crash report generation."""
    config = OverviewPlotConfiguration(
        fig_size=6,
        shift_colormap="bone_r",
        shift_auto_axes_limit=True,
        shift_axes_limit=2.0,
        theta_colormap="twilight_shifted",
    )
    points = DataFrame(
        {
            "x0": [1, 2, 3],
            "y0": [1, 2, 3],
            "dx": [0.0, 0.0, 0.0],
            "dy": [0.0, 0.0, 0.0],
            "radial error": [0.0, 0.0, 0.0],
            "angle": [0.0, 0.0, 0.0],
        }
    )

    plot = OverviewPlot(config, _mock_image(), _mock_image(), points, prefix=None)
    output_file = tmp_path / "overview.png"
    plot.plot(output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0
