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


"""KARIOS entry point module."""
import logging
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

from accuracy_analysis.accuracy_statistics import GeometricStat
from argparser import parse_args
from core.configuration import Configuration
from core.image import GdalRasterImage, get_image_resolution
from core.utils import get_filename
from klt_matcher.matcher import KLT
from log import configure_logging
from report.circular_error_plot import CircularErrorPlot
from report.overview_plot import OverviewPlot
from report.row_col_shift_plot import RowColShiftPlot
from version import __version__

logger = logging.getLogger(__name__)


class MatchAndPlot:
    # pylint: disable=too-few-public-methods
    """Object that orchestrate KTL match and plot creation."""

    def __init__(self, conf: Configuration):
        """Constructor.

        Args:
            conf (Configuration): config to apply to matching and plot.
        """
        self._conf = conf
        self._klt = KLT(
            conf.klt_configuration,
            self._conf.values.gen_delta_raster,
            self._conf.values.output_directory,
        )

    def _handle_klt_results(self, results: Iterator[pd.DataFrame], csv_file: Path) -> pd.DataFrame:
        all_frame = pd.DataFrame()
        for dataframe in results:
            if not csv_file.exists():
                logger.info("Write to csv %s", str(csv_file))
                dataframe.to_csv(csv_file, sep=";", index=False)
            else:
                logger.info("Append to csv %s", str(csv_file))
                dataframe.to_csv(csv_file, mode="a", sep=";", index=False, header=False)

            all_frame = pd.concat([all_frame, dataframe])

        return all_frame

    def _check_output_dir(self):
        out_dir_path = Path(self._conf.values.output_directory)
        if not out_dir_path.exists():
            out_dir_path.mkdir(parents=True)
        else:
            logger.warning(
                "Output dir %s already exists, some files could be overridden.",
                out_dir_path,
            )

    def _compute_stats(
        self,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        points: pd.DataFrame,
        mask: GdalRasterImage,
    ) -> GeometricStat:
        # prepare stats - select only points above confidence threshold
        acc_config = self._conf.accuracy_analysis_configuration
        stats = GeometricStat(acc_config, points, monitored_image.have_pixel_resolution())

        # compute number of valid pixels considering mask if any
        masked_image = monitored_image.array
        if mask is not None:
            masked_image = np.copy(monitored_image.array)
            masked_image[mask.array == 0] = 0

        nb_valid_pixel = np.count_nonzero(masked_image)
        logger.info("NB of valid px %s", nb_valid_pixel)
        logger.info("NB of total px %s", monitored_image.x_size * monitored_image.y_size)

        # compute stats, log and save to file
        stats.compute_stats(nb_valid_pixel)
        stats.display_results()

        stats.update_statistic_file(
            reference_image.file_name,
            monitored_image.file_name,
            os.path.join(self._conf.values.output_directory, "correl_res.txt"),
        )

        return stats

    def _do_klt(
        self,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        mask: GdalRasterImage,
        csv_file: str,
    ):
        dataframe_gen = self._klt.match(monitored_image, reference_image, mask)
        points = self._handle_klt_results(dataframe_gen, csv_file)
        return points

    def _get_points(
        self,
        resume: bool,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        mask: GdalRasterImage,
    ) -> pd.DataFrame:
        # pylint: disable=too-many-arguments
        """Get points by running KLT or reading CSV file"""

        filename = f"KLT_matcher_{get_filename(monitored_image.filepath)}_{get_filename(reference_image.filepath)}"
        csv_file = Path(os.path.join(self._conf.values.output_directory, f"{filename}.csv"))

        # run matcher:
        if not resume:
            if csv_file.exists():
                logger.warning("CSV file exists, will overwrite it: %s", str(csv_file))
                csv_file.unlink()
            points = self._do_klt(monitored_image, reference_image, mask, csv_file)
        elif not csv_file.exists():
            logger.warning("Cannot resume, CSV file missing, create it : %s", str(csv_file))
            points = self._do_klt(monitored_image, reference_image, mask, csv_file)
        else:
            logger.info("Load CSV : %s", str(csv_file))
            points = pd.read_csv(csv_file, sep=";", index_col=False)

        return points

    def _create_mask(self, points: pd.DataFrame, monitored_image: GdalRasterImage):
        logger.info("Create mask")
        # Credits Jérôme
        x_index = points["x0"].to_numpy().astype(int)
        y_index = points["y0"].to_numpy().astype(int)
        final_mask = np.zeros([monitored_image.y_size, monitored_image.x_size], dtype=np.uint8)
        final_mask[y_index, x_index] = 1
        monitored_image.to_raster(
            os.path.join(self._conf.values.output_directory, "kp_mask.tif"), final_mask
        )
        logger.info("Mask created")

    def _create_intermediate_raster(self, points: pd.DataFrame, monitored_image: GdalRasterImage):
        logger.info("Create intermediate product")

        x_index = points["x0"].to_numpy().astype(int)
        y_index = points["y0"].to_numpy().astype(int)

        dx_band_array = np.full(
            [monitored_image.y_size, monitored_image.x_size], np.nan, dtype=float
        )
        dy_band_array = np.full(
            [monitored_image.y_size, monitored_image.x_size], np.nan, dtype=float
        )

        dx_band_array[y_index, x_index] = points["dx"]
        dy_band_array[y_index, x_index] = points["dy"]

        monitored_image.to_raster(
            os.path.join(self._conf.values.output_directory, "kp_delta.tif"),
            [dx_band_array, dy_band_array],
            gdal.GDT_Float32,
        )

        logger.info("Intermediate product created")

    def _plot_overview(
        self,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        points: pd.DataFrame,
    ):
        # plot overview
        overview_plot = OverviewPlot(
            self._conf.overview_plot_configuration,
            monitored_image,
            reference_image,
            points,
            self._conf.values.title_prefix,
        )
        overview_poster_path = Path(
            os.path.join(self._conf.values.output_directory, "01_overview.png")
        )
        overview_plot.plot(overview_poster_path)

    def _plot_mean_profiles(
        self,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        points: pd.DataFrame,
    ):
        # plot dx and dy mean profiles:
        row_col_shift_plot = RowColShiftPlot(
            self._conf.shift_plot_configuration,
            monitored_image,
            reference_image,
            points,
            "dx",
            self._conf.values.title_prefix,
        )

        # plot dx
        dx_poster_path = Path(os.path.join(self._conf.values.output_directory, "02_dx.png"))
        row_col_shift_plot.plot(dx_poster_path)

        # plot dx and dy mean profiles:
        row_col_shift_plot = RowColShiftPlot(
            self._conf.shift_plot_configuration,
            monitored_image,
            reference_image,
            points,
            "dy",
            self._conf.values.title_prefix,
        )

        # plot dy
        dy_poster_path = Path(os.path.join(self._conf.values.output_directory, "03_dy.png"))
        row_col_shift_plot.plot(dy_poster_path)

    def _plot_ce(
        self,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        stats: GeometricStat,
    ):
        # plot CE
        ce_poster_path = Path(os.path.join(self._conf.values.output_directory, "04_ce.png"))

        monitored_image_resolution = get_image_resolution(
            monitored_image, reference_image, self._conf.values.pixel_size
        )

        circular_error_plot = CircularErrorPlot(
            self._conf.ce_plot_configuration,
            monitored_image,
            reference_image,
            stats,
            monitored_image_resolution,
            self._conf.values.title_prefix,
        )
        circular_error_plot.plot(ce_poster_path)

    def process(self, mon_file_path: str, ref_file_path: str, mask_file_path: str, resume: bool):
        """Orchestrates job to do.
        Process to matching, create plot and csv stat file.

        Args:
            mon_file_path (str): path to image to monitor
            ref_file_path (str): path to reference image used to monitor
            mask_file_path (str): path to masque to apply
            resume (bool): Resume or not previous process. if 'True', then KLT is not run
        """
        logger.info("Process %s", mon_file_path)

        # Prepare output dir
        self._check_output_dir()

        # Prepare input images
        monitored_image = GdalRasterImage(mon_file_path)
        reference_image = GdalRasterImage(ref_file_path)

        if mask_file_path is not None:
            mask = GdalRasterImage(mask_file_path)
        else:
            mask = None

        points = self._get_points(resume, monitored_image, reference_image, mask)

        if self._conf.values.gen_kp_mask:
            self._create_mask(points, monitored_image)

        if self._conf.values.gen_delta_raster:
            self._create_intermediate_raster(points, monitored_image)

        self._plot_overview(monitored_image, reference_image, points)
        self._plot_mean_profiles(monitored_image, reference_image, points)

        stats = self._compute_stats(monitored_image, reference_image, points, mask)
        self._plot_ce(monitored_image, reference_image, stats)


def main(argv: list[str]) -> int:
    """KARIOS entry point.

    Args:
      argv: list[str]: program arguments

    Returns:
      int: return core
      - 0 OK

    """
    args = parse_args(argv)
    configure_logging(args.debug, not args.no_log_file, args.log_file_path)
    logger.info("Start KARIOS %s with Python %s", __version__, sys.version)
    # set up configuration :
    conf = Configuration(args)
    # do the job
    match_and_plot = MatchAndPlot(conf)
    match_and_plot.process(args.mon, args.ref, args.mask, args.resume)

    shutil.copy(conf.values.configuration, conf.values.output_directory)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
