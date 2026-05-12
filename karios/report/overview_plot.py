# -*- coding: utf-8 -*-
# Copyright (c) 2025 Telespazio France.
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
"""Module to plot images, radial error and theta error"""
import logging

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, Series

from karios.core.configuration import OverviewPlotConfiguration
from karios.core.image import GdalRasterImage
from karios.report.commons import AbstractPlot

logger = logging.getLogger(__name__)


class OverviewPlot(AbstractPlot):
    # pylint: disable=too-few-public-methods
    """Overview plot class. It plots :
    - monitored image
    - reference image
    - monitored image radial error as "scatter" (one colored point by KP)
    - monitored image theta error as "scatter" (one colored point by KP)
    """

    def __init__(
        self,
        config: OverviewPlotConfiguration,
        mon_image: GdalRasterImage,
        ref_image: GdalRasterImage,
        points: DataFrame,
        prefix: str | None,
        mask: GdalRasterImage | None = None,
        no_values: list[int] | None = None,
    ):
        """Constructor

        Args:
            config (OverviewPlotConfiguration): plot config
            mon_image (GdalRasterImage): monitored image
            ref_image (GdalRasterImage): reference image
            points (DataFrame): KP data frame with series x0, y0, dx, dy
            prefix (str|None): figure title prefix
            mask (GdalRasterImage|None): optional mask applied to the monitored image
                when plotting. Pixels where mask == 0 are hidden.
            no_values (list[int]|None): optional list of DN values to hide in both
                monitored and reference image displays, matching the CLI --no-value filter.
        """
        super().__init__(prefix, config.fig_size)
        self._config = config
        self._mon_img = mon_image
        self._ref_img = ref_image
        self._points = points
        self._mask = mask
        self._no_values = no_values

    ####################################################
    # Abstract implementation
    #

    @property
    def _figure_title(self) -> str:
        return "Errors overview"

    def _prepare_figure(self, fig_size) -> Figure:
        # Calculate figure size
        fig_width = fig_size * 1.25
        fig_height = fig_size * 1.2

        return plt.figure(figsize=(fig_width, fig_height))

    def _plot(self):
        """Plot overview using manual positioning for precise control"""

        # Define precise positions [left, bottom, width, height] in figure coordinates

        # Header area
        header_pos = [0.05, 0.88, 0.9, 0.08]

        # First row: Monitored image and Radial error
        mon_img_pos = [0.05, 0.48, 0.4, 0.38]
        rad_err_pos = [0.5, 0.48, 0.5, 0.38]

        # Second row: Reference image and Theta error
        ref_img_pos = [0.05, 0.05, 0.4, 0.38]
        theta_err_pos = [0.5, 0.05, 0.5, 0.38]

        # Create plot axes first
        header_ax = self._figure.add_axes(header_pos)
        mon_img_ax = self._figure.add_axes(mon_img_pos)
        rad_err_ax = self._figure.add_axes(rad_err_pos)
        ref_img_ax = self._figure.add_axes(ref_img_pos)
        theta_err_ax = self._figure.add_axes(theta_err_pos)

        # Plot images and errors first
        self._setup_header(header_ax)
        mon_invalid = self._build_invalid_mask(self._mon_img, apply_user_mask=True)
        ref_invalid = self._build_invalid_mask(self._ref_img, apply_user_mask=False)
        self._plot_image(mon_img_ax, self._mon_img, "Monitored", invalid=mon_invalid)
        self._plot_image(ref_img_ax, self._ref_img, "Reference", invalid=ref_invalid)

        # Plot errors with colorbars
        self._plot_radial_error(rad_err_ax)
        self._plot_theta_error(theta_err_ax)

    ####################################################
    # Helper methods
    #

    def _setup_header(self, axes: Axes) -> None:
        """Setup the header with image names (reference under monitored)"""
        axes.axis("off")
        text = f"Monitored : {self._mon_img.file_name}\nReference : {self._ref_img.file_name}".expandtabs()
        axes.text(x=0, y=0.5, s=text, size="14", ha="left", va="center")

    def _build_invalid_mask(
        self, img: GdalRasterImage, apply_user_mask: bool
    ) -> np.ndarray | None:
        """Combine the user-provided mask and the --no-value DN filter into a
        boolean array marking pixels to hide. Returns None when there is nothing
        to hide so the caller can skip masked-array construction.
        """
        invalid = None
        if apply_user_mask and self._mask is not None:
            invalid = self._mask.array == 0
        if self._no_values:
            no_value_invalid = np.isin(img.array, self._no_values)
            invalid = no_value_invalid if invalid is None else (invalid | no_value_invalid)
        return invalid if invalid is not None and invalid.any() else None

    def _plot_image(
        self,
        axes: Axes,
        img: GdalRasterImage,
        title: str,
        invalid: np.ndarray | None = None,
    ) -> None:
        """Plot image with adaptive contrast for satellite imagery.

        Uses cumulative count cut (0.5%-99.5%) for consistent visualization.
        Pixels marked True in `invalid` are hidden and excluded from the
        contrast computation.
        """
        axes.set_title(title)

        if invalid is not None:
            valid_pixels = img.array[np.isfinite(img.array) & (img.array != 0) & ~invalid]
        else:
            valid_pixels = img.array[np.isfinite(img.array) & (img.array != 0)]

        if len(valid_pixels) > 0:
            v_min = np.percentile(valid_pixels, 0.5)
            v_max = np.percentile(valid_pixels, 99.5)
        else:
            v_min = np.nanmin(img.array)
            v_max = np.nanmax(img.array)

        logger.debug(
            "%s : min %s / %s , max %s / %s",
            img.filepath,
            np.nanmin(img.array),
            v_min,
            np.nanmax(img.array),
            v_max,
        )

        display_array = (
            np.ma.masked_array(img.array, mask=invalid) if invalid is not None else img.array
        )
        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad(color="magenta")
        axes.imshow(display_array, cmap=cmap, vmin=v_min, vmax=v_max)

    def _plot_radial_error(self, axes: Axes) -> None:
        """Plot radial error with dedicated colorbar"""
        dist = self._points["radial error"]
        logger.debug("Delta min %s / max %s", dist.min(), dist.max())

        # Calculate limits
        lim_min = 0
        if self._config.shift_auto_axes_limit:
            lim_max = dist.mean() + dist.std() * 3
            kwargs = {
                "norm": colors.TwoSlopeNorm(
                    vmin=lim_min, vcenter=(lim_max - lim_min) / 2, vmax=lim_max
                )
            }
        else:
            lim_max = self._config.shift_axes_limit
            kwargs = {"vmin": lim_min, "vmax": lim_max}

        scatter = self._create_error_scatter(
            axes, dist, "Radial Error (px)", self._config.shift_colormap, **kwargs
        )

        # Add colorbar
        self._figure.colorbar(scatter, aspect=35)

    def _plot_theta_error(self, axes: Axes) -> None:
        """Plot theta error with dedicated colorbar"""
        angles = self._points["angle"]

        scatter = self._create_error_scatter(
            axes,
            angles,
            "Angle error (deg), East direction CC",
            self._config.theta_colormap,
            vmin=-180,
            vmax=180,
        )

        # Add colorbar
        self._figure.colorbar(scatter, aspect=35)

    def _create_error_scatter(
        self, axes: Axes, values: Series, title: str, colormap: str, **kwargs
    ) -> plt.cm.ScalarMappable:
        """Create scatter plot for error visualization"""
        axes.set_title(title)
        axes.set_xlim(0, self._mon_img.x_size)
        axes.set_ylim(self._mon_img.y_size, 0)
        axes.axis("scaled")

        scatter = axes.scatter(
            self._points["x0"],
            self._points["y0"],
            c=values,
            cmap=colormap,
            s=1,
            **kwargs,
        )

        axes.grid()

        return scatter
