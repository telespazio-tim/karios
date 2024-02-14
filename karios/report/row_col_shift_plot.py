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
"""row col plot module"""
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core.configuration import ShiftPlotConfiguration
from core.image import GdalRasterImage
from matplotlib import colors
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import NDArray
from report.commons import add_logo


@dataclass
class MeanProfile:
    groups_positions: NDArray
    mean_values: pd.Series
    nb_pos: pd.Series
    values_std: pd.Series


def _mean_profile(values: pd.Series, positions: pd.Series, bin_size: int = 20) -> MeanProfile:
    """Compute mean and std values for each columns or rows.
    Possible to do stack several lines or columns (see bin_size) in order
    to increase number of points that is taken into account

    Args:
        values (Series): values
        positions (Series): values position on one axis, for example, X coordinates of the values
        bin_size (int, optional): bin size to group values by position group. Defaults to 20.

    Returns:
        MeanProfile: computed mean profiles
    """
    position_group = np.floor_divide(positions, bin_size)
    dic = {"val": values, "po": position_group}
    df = pd.DataFrame.from_dict(dic)
    group = df.groupby("po")
    groups_positions = np.uint16(group["po"].mean() * bin_size + bin_size / 2)
    nb_pos = group["val"].count()
    mean_values = group["val"].mean()
    values_std = group["val"].std()
    return MeanProfile(groups_positions, mean_values, nb_pos, values_std)


def _compute_std_min_max(profile: MeanProfile):
    y_err = profile.values_std * np.sqrt(
        1 / len(profile.groups_positions)
        + (profile.groups_positions - profile.groups_positions.mean()) ** 2
        / np.sum((profile.groups_positions - profile.groups_positions.mean()) ** 2)
    )
    std_min = profile.mean_values - y_err
    std_max = profile.mean_values + y_err
    return (std_min, std_max)


class RowColShiftPlot:
    # pylint: disable=too-few-public-methods
    """Class to row/col shift plot image"""

    def __init__(
        self,
        conf: ShiftPlotConfiguration,
        mon_img: GdalRasterImage,
        ref_img: GdalRasterImage,
        points,
    ):
        self._config = conf
        self._x_img_size = mon_img.x_size
        self._y_img_size = mon_img.y_size
        self._mon_img = mon_img
        self._ref_img = ref_img
        self._points = points

    def _plot_scatter(self, ax_scatter: Axes, dim):
        # scatter plot global dim shift
        kwargs = {
            "vmin": self._config.scatter_min_limit,
            "vmax": self._config.scatter_max_limit,
        }

        if self._config.scatter_auto_limit:
            lim_min = self._points[dim].mean() - self._points[dim].std() * 3
            lim_max = self._points[dim].mean() + self._points[dim].std() * 3
            if lim_max > 0 > lim_min:
                kwargs = {"norm": colors.TwoSlopeNorm(vmin=lim_min, vcenter=0.0, vmax=lim_max)}
            elif lim_min > 0:
                # center on lim_min
                kwargs = {"norm": colors.TwoSlopeNorm(vmin=0, vcenter=lim_min, vmax=lim_max)}
            elif lim_max < 0:
                # center on lim_max
                kwargs = {"norm": colors.TwoSlopeNorm(vmin=lim_min, vcenter=lim_max, vmax=0)}

        ax_scatter.set_xlim(0, self._x_img_size)
        ax_scatter.set_ylim(self._y_img_size, 0)

        # the scatter plot:
        scatter = ax_scatter.scatter(
            self._points["x0"],
            self._points["y0"],
            c=self._points[dim],
            cmap=self._config.scatter_colormap,
            s=1,
            **kwargs,
        )
        ax_scatter.legend(loc="upper left", title=f"Nb KP={len(self._points['x0'])}")
        ax_scatter.set_title(f"{dim} pixels shifts")

        return scatter

    def _plot_col(self, ax_col: Axes, dim: str):
        """line plot dim shift along column (horizontal line plot)

        Args:
            ax_col (Axes): matplotlib figure axes to plot in
            dim (str): name of the series of the `points` dataframe to plot (dx/dy)
        """
        # compute stats
        profile = _mean_profile(
            self._points[dim], self._points["x0"], self._config.histo_mean_bin_size
        )

        # ########################################################
        # PLOT NB KP BAR
        # set axis limit
        ax_col.set_ylim(0, max(profile.nb_pos))
        # plot it
        ax_col.bar(
            profile.groups_positions,
            profile.nb_pos,
            width=self._config.histo_mean_bin_size,
            color="grey",
            # edgecolor="grey",
            alpha=0.2,
            label="KP per col",
        )
        # show legend
        ax_col.legend(loc="upper right")

        # ########################################################
        # PLOT "dim" mean line
        # New X axis for mean line
        ax_col_mean = ax_col.twinx()
        # Y axis limit
        lim = math.modf(max(abs(profile.mean_values)))[1] + 1
        ax_col_mean.set_ylim(-lim, lim)

        # Plot it
        ax_col_mean.plot(profile.groups_positions, profile.mean_values, label=f"Mean {dim} by col")

        # PLOT "dim" STD by filling on X
        std_min, std_max = _compute_std_min_max(profile)
        ax_col_mean.fill_between(profile.groups_positions, std_min, std_max, alpha=0.3, label="std")

        # show legend of main Y axis
        ax_col_mean.legend(loc="upper left")

        # reverse axis to have mean and std to the left
        ax_col_mean.yaxis.axes.axhline(c="grey", lw=1, linestyle="--")
        ax_col_mean.yaxis.tick_left()
        ax_col.yaxis.tick_right()

    def _plot_row(self, ax_row: Axes, dim: str):
        """line plot dim shift along row (vertical line plot)

        Args:
            ax_row (Axes): matplotlib figure axes to plot in
            dim (str): name of the series of the `points` dataframe to plot (dx/dy)
        """
        # Compute stats
        profile = _mean_profile(
            self._points[dim], self._points["y0"], self._config.histo_mean_bin_size
        )

        # ########################################################
        # PLOT NB KP BAR
        # set axis limit
        ax_row.set_xlim(0, max(profile.nb_pos))
        # plot it
        # NOTE h at the end of bar for horizontal bar
        ax_row.barh(
            profile.groups_positions,
            profile.nb_pos,
            height=self._config.histo_mean_bin_size,
            color="grey",
            # edgecolor="grey",
            alpha=0.2,
            label="KP per row",
        )
        # show legend
        ax_row.legend(loc="lower right")

        # ########################################################
        # PLOT "dim" mean line
        # New axis for mean line
        ax_row_mean = ax_row.twiny()

        # Main X axis limit
        lim = math.modf(max(abs(profile.mean_values)))[1] + 1
        ax_row_mean.set_xlim(-lim, lim)

        # Plot main Y
        ax_row_mean.plot(profile.mean_values, profile.groups_positions, label=f"Mean {dim} by row")

        # PLOT "dim" STD by filling on Y
        std_min, std_max = _compute_std_min_max(profile)
        # NOTE x at the end of fill_between
        ax_row_mean.fill_betweenx(
            profile.groups_positions, std_min, std_max, alpha=0.3, label="std"
        )

        # show legend of main X axis
        ax_row_mean.legend(loc="upper right")

        # Reverse axis to have mean and std to the top
        ax_row_mean.xaxis.axes.axvline(c="grey", lw=1, linestyle="--")

        ax_row_mean.xaxis.tick_top()
        ax_row.xaxis.tick_bottom()

    def plot(self, direction: str, output_file: Path):
        """Plot for the given direction in file

        Args:
            direction (str): direction to plot (dx or dy)
            output_file (Path): path to the image output file
        """

        if not direction in ["dx", "dy"]:
            raise ValueError("Invalid direction value, expect 'dx' or 'dy'")

        # Start with a square Figure.
        fig = plt.figure(figsize=(self._config.fig_size, self._config.fig_size))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal axes and the main axes in both directions.
        grid = fig.add_gridspec(
            4,
            2,
            width_ratios=(4, 1),
            height_ratios=(0.5, 1, 4, 0.25),
            left=0.15,
            right=0.95,
            bottom=0.01,
            top=0.95,
            wspace=0.1,
            hspace=0.2,
        )

        # Create the Axes.
        ax_header = fig.add_subplot(grid[0, :])
        ax_scatter = fig.add_subplot(grid[2, 0])
        ax_col = fig.add_subplot(grid[1, 0], sharex=ax_scatter)
        ax_row = fig.add_subplot(grid[2, 1], sharey=ax_scatter)

        logo_gd = grid[3, :].subgridspec(1, 3)

        ax_header.axis("off")
        text = f"Monitored : {self._mon_img.file_name}\nReference : {self._ref_img.file_name}".expandtabs()
        ax_header.text(x=0, y=0.5, s=text, size="14", ha="left", va="center")

        # no labels
        ax_col.tick_params(axis="x", labelbottom=False)
        ax_row.tick_params(axis="y", labelleft=False)

        # Draw the scatter and line/bar plots.
        scatter = self._plot_scatter(ax_scatter, direction)
        self._plot_col(ax_col, direction)
        self._plot_row(ax_row, direction)

        # Add grid everywhere
        ax_scatter.grid()

        ax_col.grid(axis="x")
        ax_row.grid(axis="y")

        # colorbar in gridspec, thanks to this
        # https://stackoverflow.com/a/57623427
        cax = inset_axes(
            ax_scatter,  # here using axis of the scatter
            width="3%",  # width = 5% of parent_bbox width
            height="100%",
            loc="upper left",
            bbox_to_anchor=(-0.12, 0, 1, 1),
            bbox_transform=ax_scatter.transAxes,
            borderpad=0,
        )

        fig.colorbar(scatter, cax=cax, ticklocation="left")

        fig.suptitle(
            f"{direction} pixel shift Mean and STD by row/col (1 row/col = {self._config.histo_mean_bin_size}px)",
            size="16",
            y=0.98,
        )

        add_logo(fig, logo_gd)

        plt.savefig(output_file)

        plt.close()
