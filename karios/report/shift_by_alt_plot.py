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
"""plot by alt profile module"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from karios.core.image import GdalRasterImage
from karios.report.commons import AbstractPlot, mean_profile


class MeanShiftByAltitudeGroupPlot(AbstractPlot):

    # pylint: disable=too-few-public-methods
    """Class to plot shift depending altitudes image"""

    def __init__(
        self,
        conf: dict,
        mon_img: GdalRasterImage,
        ref_img: GdalRasterImage,
        dem_img: GdalRasterImage,
        points: DataFrame,
        direction: str,
        prefix: str | None,
        dem_desc: str | None,
        mini,
        maxi,
    ):
        """Constructor

        Args:
            conf (dict): plot config
            mon_img (GdalRasterImage): monitored image
            ref_img (GdalRasterImage): reference image
            dem_img (GdalRasterImage): dem image
            points (DataFrame): KP data frame with series x0, y0, dx, dy
            direction (str): points dimension/direction to plot, should be "dx" or "dy"
            prefix (str | None): figure title prefix
            dem_desc (str | None): DEM description, added to dem filename to output plot report
            mini
            maxi
        """
        super().__init__(prefix, conf.fig_size)
        self._config = conf
        self._x_img_res = mon_img.x_res
        self._mon_img = mon_img
        self._ref_img = ref_img
        self._dem_img = dem_img
        self._points = points
        self._direction = (
            direction  # name of the series of the `points` dataframe to plot (dx/dy/re)
        )
        self._dem_desc = dem_desc
        self._mini = mini
        self._maxi = maxi

    ####################################################
    # Abstract implementation
    #

    @property
    def _figure_title(self) -> str:
        return f"{self._direction} shift by altitudes (alt bin size: {self._config.histo_mean_bin_size}m, image resolution {self._x_img_res})"

    def _prepare_figure(self, fig_size) -> Figure:
        # Start with a square Figure.
        return plt.figure(figsize=(fig_size, fig_size))

    def _plot(self):
        """Plot"""
        if not self._direction in ["dx", "dy", "radial error"]:
            raise ValueError("Invalid direction value, expect 'dx' or 'dy'")

        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal axes and the main axes in both directions.
        grid = self._figure.add_gridspec(
            2,
            1,
            #     width_ratios=(4, 1),
            height_ratios=(0.3, 4),
            #     left=0.15,
            #     right=0.95,
            bottom=0.05,
            top=0.95,
            wspace=0.1,
            hspace=0.2,
        )

        # Create the Axes.
        ax_header = self._figure.add_subplot(grid[0, :])
        ax_plot = self._figure.add_subplot(grid[1, :])

        dem_title = self._dem_img.file_name
        if self._dem_desc:
            dem_title += f" ({self._dem_desc})"
        # set header
        ax_header.axis("off")
        text = f"Monitored : {self._mon_img.file_name}\nReference : {self._ref_img.file_name}\nDEM : {dem_title}".expandtabs()
        ax_header.text(x=0, y=0.5, s=text, size="14", ha="left", va="center")

        # do main plot
        self._plot_alt(ax_plot)

    ####################################################
    # Local implementation
    #

    def _plot_alt(self, axis: Axes):
        dim = self._direction

        profile = mean_profile(
            self._points[dim] * self._x_img_res,
            self._points["alt"],
            self._config.histo_mean_bin_size,
        )

        # plot altitude with bar by grouping them using histo_mean_bin_size
        b = axis.bar(
            profile.groups_positions,
            profile.nb_pos,
            width=self._config.histo_mean_bin_size,
            color="grey",
            edgecolor="grey",
            alpha=0.2,
            label="KP per alt group",
        )
        axis.bar_label(b, label_type="center", rotation=90)

        axis.legend(loc="upper left")
        axis.set_xlabel("Altitude (m)", loc="center")
        axis.set_ylabel("NB KP", loc="center")

        axis_right = axis.twinx()

        arr = []
        for g in profile.grouped_values:
            arr.append(g[1].to_numpy())

        axis_right.boxplot(
            arr,
            positions=profile.groups_positions,
            showfliers=self._config.show_fliers,
            # showmeans=True,
            # meanline=True,
            manage_ticks=False,
            widths=self._config.histo_mean_bin_size * 0.8,
            # notch=True,
            # patch_artist=True,
            # boxprops={"alpha": 0.3, "color": "green", , "facecolor": "green"},
            boxprops={"alpha": 0.5, "color": "green"},
            label=f"Median {dim} deviation",
        )

        axis_right.plot(
            profile.groups_positions,
            profile.mean_values,
            ".-",
            label=f"Mean {dim} deviation",
        )

        std_min, std_max = profile.compute_std_min_max()
        axis_right.fill_between(
            profile.groups_positions,
            std_min,
            std_max,
            alpha=0.2,
            label=f"STD {dim} deviation",
        )

        axis_right.plot(
            profile.groups_positions,
            np.sqrt(profile.mean_values**2 + profile.values_std**2),
            ".-",
            label=f"RMSE of {dim} deviation",
        )

        axis_right.set_ylim([self._mini * self._x_img_res, self._maxi * self._x_img_res])

        # GRID
        axis_right.grid(True)
        # OR
        # plt.axhline(c="grey", lw=1, linestyle="--")
        # plt.axhline(1, c="grey", lw=0.5, linestyle="--")
        # plt.axhline(-1, c="grey", lw=0.5, linestyle="--")

        # plt.legend(loc="upper right")
        axis_right.set_ylabel("Deviation (px)", loc="center")
        axis_right.legend(loc="upper right")
