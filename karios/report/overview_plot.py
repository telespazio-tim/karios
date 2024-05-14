# -*- coding: utf-8 -*-
# Copyright (c) 2024 Telespazio France.
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
from pathlib import Path

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import DataFrame, Series

from core.configuration import OverviewPlotConfiguration
from core.image import GdalRasterImage
from report.commons import add_logo

logger = logging.getLogger()


def theta_deg(y: float, x: float):
    return np.degrees(np.arctan2(y, x))


v_theta_deg = np.vectorize(theta_deg)


class OverviewPlot:
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
    ):
        """Constructor

        Args:
            config (OverviewPlotConfiguration): plot config
            mon_image (GdalRasterImage): monitored image
            ref_image (GdalRasterImage): reference image
            points (DataFrame): KP data frame with series x0, y0, dx, dy
        """
        self._config = config
        self._mon_img = mon_image
        self._ref_img = ref_image
        self._points = points

        # init figure
        self._figure = plt.figure(figsize=(self._config.fig_size, self._config.fig_size))
        self._figure.suptitle("Errors overview", y=0.98, size="16")

    def _add_image(self, axes: Axes, img: GdalRasterImage, title: str):
        """Add image overview to figure in the axes

        Args:
            axes (Axes): axes to put image
            img (GdalRasterImage): image to overview
            title (str): pot title
        """
        axes.set_title(title)

        # Attempt to adapt dynamic
        mean = np.mean(img.array, where=img.array != 0)
        std = np.std(img.array, where=img.array != 0)
        v_min = mean - 4 * std
        v_max = mean + 4 * std

        logger.debug(
            "%s : min %s / %s , max %s / %s",
            img.filepath,
            np.nanmin(img.array),
            v_min,
            np.nanmax(img.array),
            v_max,
        )

        axes.imshow(img.array, cmap="gray", vmin=v_min, vmax=v_max)  # , vmin=0, vmax=800)

        # new_img = cv2.equalizeHist(img.array.astype(np.uint8))
        # logger.info("%s : min %s / %s , max %s/ %s", img.filepath, np.min(img.array), np.min(new_img), np.max(img.array), np.max(new_img))
        # axes.imshow(new_img, cmap="gray")

    def _plot_error(
        self,
        axes: Axes,
        values: Series,
        title: str,
        color_map: str,
        limit: [float, float],
        div_norm: bool = False,
        norm_center: float = 0.0,
    ):
        """Scatter plot having monitored image size

        Args:
            axes (Axes): where to plot the scatter in the figure
            values (Series): values to plot
            title (str): plot title
            color_map (str): colormap to apply
            limit (float, float]): min and max limit
            div_norm (bool, optional): normalise or not the colormap with TwoSlopeNorm. Defaults to False.
            norm_center (float, optional): if `div_norm`, the TwoSlopeNorm center. Defaults to 0.0.
        """
        axes.set_title(title)
        axes.set_xlim(0, self._mon_img.x_size)
        axes.set_ylim(self._mon_img.y_size, 0)
        axes.axis("scaled")
        kwargs = {
            "vmin": limit[0],
            "vmax": limit[1],
        }

        if div_norm:
            kwargs = {
                "norm": colors.TwoSlopeNorm(vmin=limit[0], vcenter=norm_center, vmax=limit[1]),
            }

        scatter = axes.scatter(
            self._points["x0"], self._points["y0"], c=values, cmap=color_map, s=1, **kwargs
        )
        axes.grid()
        # colorbar in gridspec, thanks to this
        # https://stackoverflow.com/a/57623427
        cax = inset_axes(
            axes,  # here using axis of the scatter
            width="3%",  # width = 5% of parent_bbox width
            height="100%",
            loc="upper left",
            bbox_to_anchor=(1.02, 0, 1, 1),
            bbox_transform=axes.transAxes,
            borderpad=0,
        )

        self._figure.colorbar(scatter, cax=cax)

    def plot(self, out_image_file: Path):
        """Plot overview in file

        Args:
            out_image_file (Path): destination file path
        """

        grid = self._figure.add_gridspec(
            4,
            2,
            width_ratios=(2, 2),
            height_ratios=(0.5, 2, 2, 0.25),
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.95,
            wspace=0.05,
            hspace=0.2,
        )

        header_ax = self._figure.add_subplot(grid[0, :])
        mon_img_ax = self._figure.add_subplot(grid[1, 0])
        ref_img_ax = self._figure.add_subplot(grid[2, 0])
        rad_err_ax = self._figure.add_subplot(grid[1, 1])
        theta_err_ax = self._figure.add_subplot(grid[2, 1])
        logo_gd = grid[3, :].subgridspec(1, 3)

        header_ax.axis("off")
        text = f"Monitored : {self._mon_img.file_name}\nReference : {self._ref_img.file_name}".expandtabs()
        header_ax.text(x=0.1, y=0.5, s=text, size="14", ha="left", va="center")

        # /////////////////////////////
        # plot images
        self._add_image(mon_img_ax, self._mon_img, "Monitored")
        self._add_image(ref_img_ax, self._ref_img, "Reference")

        # /////////////////////////////
        # plot radial error
        dist = np.sqrt(self._points["dx"] ** 2 + self._points["dy"] ** 2)
        logger.debug("Delta min %s / max %s", np.min(dist), np.min(dist))

        # set axes limit
        lim_min = 0
        if self._config.shift_auto_axes_limit:
            # lim_max = np.max(dist)
            lim_max = dist.mean() + dist.std() * 3
        else:
            lim_max = self._config.shift_axes_limit

        self._plot_error(
            rad_err_ax,
            dist,
            "Radial Error (px)",
            self._config.shift_colormap,
            [lim_min, lim_max],
            div_norm=self._config.shift_auto_axes_limit,
            norm_center=(lim_max - lim_min) / 2,
        )

        # /////////////////////////////
        # plot theta error
        angles = v_theta_deg(self._points["dy"], self._points["dx"])
        self._plot_error(
            theta_err_ax,
            angles,
            "Angle error (deg), East direction CC",
            self._config.theta_colormap,
            [-180, 180],
        )

        add_logo(self._figure, logo_gd)

        # serialize plot
        plt.savefig(out_image_file)
        plt.close()
