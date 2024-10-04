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
"""circular error plot module"""

import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats as sp_stats
from scipy.interpolate import interpn

from accuracy_analysis.accuracy_statistics import GeometricStat
from core.configuration import CEPlotConfiguration
from core.image import GdalRasterImage
from report.commons import AbstractPlot, add_logo

logger = logging.getLogger()


def _rmse(mean, std, img_res=None):
    if img_res is not None:
        _mean = mean * img_res
        _std = std * img_res
    else:
        _mean = mean
        _std = std

    return np.sqrt(_mean * _mean + _std * _std)


class CircularErrorPlot(AbstractPlot):
    # pylint: disable=too-few-public-methods
    """Class to create circular error plot image. It plots :
    - CE scatter
    - E/N x/y displacement histogram
    - radial error
    - statistics text box
    """

    def __init__(
        self,
        conf: CEPlotConfiguration,
        mon_image: GdalRasterImage,
        ref_image: GdalRasterImage,
        stats: GeometricStat,
        img_res: float | None,
        prefix: str | None,
    ):
        """Constructor

        Args:
            conf (CEPlotConfiguration): plot config
            mon_image (GdalRasterImage): image to match
            ref_image (GdalRasterImage): reference image
            stats (GeometricStat): statistics to plot
            img_res (float | None): image resolution to apply to statistics as factor.
                if None, consider pixel with value 1, otherwise, consider meter.
            prefix (str|None): figure title prefix
        """
        super().__init__(prefix, conf.fig_size)
        self._conf = conf
        self._mon_img = mon_image
        self._ref_img = ref_image
        self._stats = stats
        self._img_res = img_res

        self._unit = "meter"
        self._short_unit = "m"
        self._x_scatter_label = "Easting displacement (meter)"
        self._y_scatter_label = "Northing displacement (meter)"

        self._forced_image_resolution = False
        if self._img_res is not None and not self._mon_img.have_pixel_resolution():
            self._forced_image_resolution = True

        if self._img_res is None:
            self._img_res = 1.0
            self._unit = "pixel"
            self._short_unit = "px"
            self._x_scatter_label = "Row displacement (pixel)"
            self._y_scatter_label = "Line displacement (pixel)"

    ####################################################
    # Abstract implementation
    #

    @property
    def _figure_title(self) -> str:
        return "Geometric Error distribution"

    def _prepare_figure(self, fig_size) -> Figure:
        return plt.figure(figsize=(fig_size * 5 / 3, fig_size * 1.2))

    def _plot(self):

        grid = self._figure.add_gridspec(
            4,
            3,
            width_ratios=(4, 4, 4),
            height_ratios=(0.5, 4, 4, 0.5),
            left=0.15,
            right=0.9,
            bottom=0.02,
            top=0.9,
            wspace=0.1,
            hspace=0.3,
        )

        # Create the Axes for each plot
        ax_header = self._figure.add_subplot(grid[0, :])
        ax_scatter = self._figure.add_subplot(grid[2, 0])
        ax_col = self._figure.add_subplot(grid[1, 0], sharex=ax_scatter)
        ax_row = self._figure.add_subplot(grid[2, 1], sharey=ax_scatter)
        ax_text = self._figure.add_subplot(grid[1, 1])
        ax_ce = self._figure.add_subplot(grid[1, 2])
        logo_gd = grid[3, :].subgridspec(1, 3)

        # no labels
        ax_col.tick_params(axis="x", labelbottom=False)
        ax_row.tick_params(axis="y", labelleft=False)

        # Add input images name and disclaimer if needed
        self._set_header(ax_header)

        #  Plot Scatter :
        scatter_plot = self._ce_scatter(ax_scatter)

        # ///////////////////////////////////////
        # plot col
        self._hist_vector(ax_col, self._stats.v_y_th, "y")

        # ///////////////////////////////////////
        # plot row
        self._hist_vector(
            ax_row,
            self._stats.v_x_th,
            "x",
            orientation="horizontal",
        )

        # TODO : add cumul (CDF)
        self._radial_error_plot(ax_ce)

        self._text_box(ax_text)

        # colorbar in gridspec, thanks to this
        # https://stackoverflow.com/a/57623427
        cax = inset_axes(
            ax_scatter,  # here using axis of the scatter
            width="3%",  # width = 5% of parent_bbox width
            height="100%",
            loc="upper left",
            bbox_to_anchor=(-0.3, 0, 1, 1),
            bbox_transform=ax_scatter.transAxes,
            borderpad=0,
        )

        self._figure.colorbar(scatter_plot, cax=cax, ticklocation="left")

        add_logo(self._figure, logo_gd)

    ####################################################
    # Local implementation
    #

    def _compute_histogram(self, vect, direction):
        # The number of Bins corresponding to 0.1 pixel :
        # pas = 0.1

        v_max = np.max([np.abs(np.max(vect)), np.abs(np.min(vect))])
        logger.info("value %s", str(2 * v_max))
        # number_of_bin = int(((2 * v + 1) / pas))
        logger.info(" pixel size                      : %s %s", self._img_res, self._short_unit)
        logger.info(
            " Bin range (%s)                   : [-%s , %s]",
            self._short_unit,
            v_max * self._img_res,
            v_max * self._img_res,
        )
        logger.info(
            " Optimal bin width computation, quare root (of data size) estimator method (sqrt)  "
        )

        # sqrt     Square root (of data size) estimator, used by Excel
        # and other programs for its speed and simplicity.

        # bins , return the bins edge, (length(hist)+1)
        (hist, bins) = np.histogram(
            vect * self._img_res,
            bins="sqrt",
            range=(-v_max * self._img_res, v_max * self._img_res),
        )

        npixtotal = np.sum(hist)

        # Get text block to log
        out_str_list = [self._stats.get_string_block(self._img_res, direction=direction)]
        out_str_list.append(f"Total Pixels: {str(npixtotal)}")
        out_str_list.append(f"Nbr of bins : {str(len(hist))}")

        # Normal Test :
        k2, p = sp_stats.normaltest(hist)
        alpha = 1e-3
        if p < alpha:  # null hypothesis: hist comes from a normal distribution
            # logger.info("The null hypothesis can be rejected, p value : %s", p)
            out_str_list.append("Normal Test : rejected")
        else:
            # logger.info("The null hypothesis cannot be rejected, p value : %s", p)
            out_str_list.append("Normal Test : not rejected")

        logger.info("\n".join(out_str_list))

        return (hist, bins)

    def _set_header(self, axes: Axes):
        axes.axis("off")
        text = f"Monitored : {self._mon_img.file_name}\nReference : {self._ref_img.file_name}".expandtabs()
        axes.text(x=0, y=0, s=text, size="12", ha="left", va="top")

        # Add disclaimer
        if not self._mon_img.get_epsg() or (self._mon_img.get_epsg() != self._ref_img.get_epsg()):
            axes.text(
                x=0.5,
                y=0.5,
                s="\n".join(
                    [
                        "Disclaimer: ",
                        "Planimetric accuracy results may not be relevant in the case of",
                        "input images provided without, or not identical map projection",
                    ]
                ),
                # color="orange",
                size="14",
                ha="center",
                va="bottom",
                bbox={"facecolor": "none", "edgecolor": "red", "pad": 5.0},
            )

    def _ce_scatter(self, axes: Axes):
        # Computation CE90 2D :
        ce_90 = self._stats.compute_percentile(0.9, self._img_res)
        x = self._stats.v_x_th * self._img_res
        y = self._stats.v_y_th * self._img_res

        # Plot all values in the graphic both flatten arrays :
        # Determine axis values :
        v1 = np.max(np.abs(x))
        v2 = np.max(np.abs(y))

        plot_limit = np.max([v1, v2])

        bins = 10
        data, x_e, y_e = np.histogram2d(x, y, bins=bins)

        z = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([x, y]).T,
            method="splinef2d",
            bounds_error=False,
        )

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        # PLOT scatter
        scatter = axes.scatter(y, x, c=z, cmap=self._conf.ce_scatter_colormap)

        # Plot in the graphic Cicrular error circle :
        u = range(0, 110, 1)
        theta = (np.array(u) / 100.0) * 2 * np.pi
        x_ce = ce_90 * np.cos(theta)
        y_ce = ce_90 * np.sin(theta)
        axes.plot(x_ce, y_ce, "-")

        # configure axis
        axes.set_xlabel(self._x_scatter_label)
        axes.set_ylabel(self._y_scatter_label)

        axes.set_xlim([-plot_limit, plot_limit])
        axes.set_ylim([-plot_limit, plot_limit])

        axes.grid()

        axes.set_title("Circular Error Plot @ 90 percentile", fontsize=11)

        return scatter

    def _hist_vector(self, axes: Axes, vect, direction, orientation="vertical"):
        (hist, bins) = self._compute_histogram(vect, direction)

        # starting from bin edge compute the center of each bin
        # center = (bins[:-1] + bins[1:]) / 2
        # http://www.python-simple.com/python-matplotlib/barplot.php
        # axes.bar(
        #     center,
        #     hist,
        #     # width=pas / 10.0,
        #     color="blue",
        #     # edgecolor="blue",
        #     # linewidth=0.5,
        #     label="Count",
        # )

        axes.hist(
            bins[:-1],
            bins,
            weights=hist,
            orientation=orientation,
            edgecolor="grey",
            # color="tab:blue",
            color="grey",
            label="Count",
            alpha=0.4,
        )

        # Gaussian curve
        npixtotal = np.sum(hist)
        mu = np.mean(vect * self._img_res)
        sigma = np.std(vect * self._img_res)
        n = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((bins - mu) ** 2) / (2 * sigma**2))

        x_val = bins
        y_val = n / (np.sum(n)) * npixtotal

        # swap x/y for horizontal
        if orientation == "horizontal":
            temp = x_val
            x_val = y_val
            y_val = temp

        axes.plot(
            x_val,
            y_val,
            linewidth=1.5,
            # color="tab:red",
            label=f"Normal (mean={mu:.2f}, sigma={sigma:.2f})",
        )

        axes.grid()
        axes.legend()

    def _radial_error_plot(self, axes: Axes):
        x = self._stats.v_x_th * self._img_res
        y = self._stats.v_y_th * self._img_res

        v = np.sort(np.sqrt(x * x + y * y))

        x = np.linspace(0, 100, v.shape[0])

        ce_90 = self._stats.compute_percentile(0.9, self._img_res)
        ce_95 = self._stats.compute_percentile(0.95, self._img_res)

        axes.plot(x, v, label="Radial Error")
        axes.plot(90, ce_90, "+", label=f"ce 90 : {ce_90:.2f} {self._short_unit}")
        axes.plot(95, ce_95, "o", label=f"ce 95 : {ce_95:.2f} {self._short_unit}")

        axes.grid()
        axes.set_xlabel("Sample percentage")
        axes.set_ylabel("Radial Error")
        axes.legend()

    def _text_box(self, axes):
        axes.axis("off")

        y_rmse = _rmse(self._stats.mean_y, self._stats.std_y, self._img_res)
        x_rmse = _rmse(self._stats.mean_x, self._stats.std_x, self._img_res)

        text_list = [
            f"Total Number of Key Point : {self._stats.sample_pixel}",
            f"Confidence value : {self._stats.confidence:.2f}",
            f"Percentage of Confident Pixels : {self._stats.percentage_of_pixel:.2f}%",
            "",
            f"{self._x_scatter_label}:",
            f"\tMin : {self._stats.min_y*self._img_res:.2f} {self._short_unit}",
            f"\tMax : {self._stats.max_y*self._img_res:.2f} {self._short_unit}",
            f"\tMean : {self._stats.mean_y*self._img_res:.2f} {self._short_unit}",
            f"\tSigma : {self._stats.std_y*self._img_res:.2f} {self._short_unit}",
            f"\tRMSE : {y_rmse:.2f} {self._short_unit}",
            "",
            f"{self._y_scatter_label}:",
            f"\tMin : {self._stats.min_x*self._img_res:.2f} {self._short_unit}",
            f"\tMax : {self._stats.max_x*self._img_res:.2f} {self._short_unit}",
            f"\tMean : {self._stats.mean_x*self._img_res:.2f} {self._short_unit}",
            f"\tSigma : {self._stats.std_x*self._img_res:.2f} {self._short_unit}",
            f"\tRMSE : {x_rmse:.2f} {self._short_unit}",
            "",
            f"Global RMSE : {_rmse(x_rmse, y_rmse):.2f} {self._short_unit}",
            f"CE @90 the percentile : {self._stats.compute_percentile(0.9, self._img_res):.2f} {self._short_unit}",
            f"CE @95 the percentile : {self._stats.compute_percentile(0.95, self._img_res):.2f} {self._short_unit}",
        ]

        epsg = self._mon_img.get_epsg()
        if not self._forced_image_resolution and epsg is not None:
            text_list.insert(
                3,
                f"Pixel size : {self._img_res} {self._short_unit} \tEPSG: {self._mon_img.get_epsg()}".expandtabs(),
            )
        elif self._forced_image_resolution:
            text_list.insert(3, f"Pixel size : {self._img_res} {self._short_unit}")

        text = "\n".join(text_list).expandtabs()

        logger.info(text)
        axes.text(x=0, y=0, s=text)
