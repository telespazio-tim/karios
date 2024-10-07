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
import math
import os
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

from accuracy_analysis.accuracy_statistics import GeometricStat
from argparser import KariosArgumentParser
from core.configuration import Configuration
from core.errors import KariosException
from core.image import GdalRasterImage, get_image_resolution, shift_image
from core.utils import get_filename
from log import configure_logging
from matcher.klt import KLT
from matcher.large_offset import LargeOffsetMatcher
from report.circular_error_plot import CircularErrorPlot
from report.overview_plot import OverviewPlot
from report.product_generator import ProductGenerator
from report.shift_by_alt_plot import MeanShiftByAltitudeGroupPlot
from report.shift_by_row_col_plot import MeanShiftByRowColGroupPlot
from version import __version__

gdal.UseExceptions()
logger = logging.getLogger(__name__)


@dataclass
class ShiftedImage:
    """Describe a shifted image"""

    image: GdalRasterImage
    """shifted image"""
    x_offset: int
    """X/col offset compared to original image in pixel"""
    y_offset: int
    """Y/row offset compared to original image in pixel"""


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
            dataframe["radial error"] = np.sqrt(dataframe["dx"] ** 2 + dataframe["dy"] ** 2)
            dataframe["angle"] = np.degrees(np.arctan2(dataframe["dy"], dataframe["dx"]))
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
        mask: GdalRasterImage | None,
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

    def _compute_delta(
        self,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        mask: GdalRasterImage | None,
        csv_file: Path,
    ):
        dataframe_gen = self._klt.match(monitored_image, reference_image, mask)
        points = self._handle_klt_results(dataframe_gen, csv_file)
        return points

    def _get_points(
        self,
        resume: bool,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        mask: GdalRasterImage | None,
    ) -> pd.DataFrame:
        """Get points by running KLT or reading CSV file"""

        filename = f"KLT_matcher_{get_filename(monitored_image.filepath)}_{get_filename(reference_image.filepath)}"
        csv_file = Path(os.path.join(self._conf.values.output_directory, f"{filename}.csv"))

        # run matcher:
        if not resume:
            if csv_file.exists():
                logger.warning("CSV file exists, will overwrite it: %s", str(csv_file))
                csv_file.unlink()
            points = self._compute_delta(monitored_image, reference_image, mask, csv_file)
        elif not csv_file.exists():
            logger.warning("Cannot resume, CSV file missing, create it : %s", str(csv_file))
            points = self._compute_delta(monitored_image, reference_image, mask, csv_file)
        else:
            logger.info("Load CSV : %s", str(csv_file))
            points = pd.read_csv(csv_file, sep=";", index_col=False)

        return points

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

    def _plot_mean_shift_by_row_col_group(
        self,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        points: pd.DataFrame,
    ):
        # plot dx mean profiles:
        shift_by_row_col_plot = MeanShiftByRowColGroupPlot(
            self._conf.shift_plot_configuration,
            monitored_image,
            reference_image,
            points,
            "dx",
            self._conf.values.title_prefix,
        )
        dx_poster_path = Path(os.path.join(self._conf.values.output_directory, "02_dx.png"))
        shift_by_row_col_plot.plot(dx_poster_path)

        # plot dy mean profiles:
        shift_by_row_col_plot = MeanShiftByRowColGroupPlot(
            self._conf.shift_plot_configuration,
            monitored_image,
            reference_image,
            points,
            "dy",
            self._conf.values.title_prefix,
        )
        dy_poster_path = Path(os.path.join(self._conf.values.output_directory, "03_dy.png"))
        shift_by_row_col_plot.plot(dy_poster_path)

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

    def _plot_dem(
        self,
        dem: GdalRasterImage,
        points: pd.DataFrame,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
    ):
        x_index = points["x0"].to_numpy().astype(int)
        y_index = points["y0"].to_numpy().astype(int)
        points["alt"] = dem.array[y_index, x_index]

        conf = self._conf.dem_plot_configuration

        # compute min and max for shift axis to use the same for each plots
        maxi = math.ceil(max(points["dx"].max(), points["dy"].max(), points["radial error"].max()))
        mini = math.floor(min(points["dx"].min(), points["dy"].min(), points["radial error"].min()))

        for dimension in ["dx", "dy", "radial error"]:

            report = MeanShiftByAltitudeGroupPlot(
                conf,
                monitored_image,
                reference_image,
                dem,
                points,
                dimension,
                self._conf.values.title_prefix,
                self._conf.values.dem_description,
                mini,
                maxi,
            )
            poster_path = Path(
                os.path.join(
                    self._conf.values.output_directory, f"dem_{dimension}.png".replace(" ", "_")
                )
            )
            report.plot(poster_path)

    def _get_dem(self, reference_image: GdalRasterImage) -> GdalRasterImage | None:
        """Get DEM as GdalRasterImage and check its compatibility with reference image.
        Notice that DEM path is in this object config.

        Args:
            reference_image (GdalRasterImage): reference image

        Raises:
            KariosException: if DEM is not compatible with reference image

        Returns:
            GdalRasterImage | None: DEM if present in conf and compatible with reference image.
        """
        if self._conf.values.dem_file_path:
            dem = GdalRasterImage(self._conf.values.dem_file_path)
            if not dem.is_compatible_with(reference_image):
                raise KariosException(
                    f"""DEM geo info not compatible with reference image shape or resolution:
                * DEM image : {dem.image_information}
                * Reference image : {reference_image.image_information}
                """
                )
        else:
            dem = None

        return dem

    @staticmethod
    def _get_images(
        ref_file_path: str, mon_file_path: str
    ) -> tuple[GdalRasterImage, GdalRasterImage]:
        """Build GdalRasterImage of images denoted by their file path and check compatibility.

        Args:
            ref_file_path (str): file path to reference image
            mon_file_path (str): file path to monitored image

        Raises:
            KariosException: if monitored image is not compatible with reference image

        Returns:
            tuple[GdalRasterImage, GdalRasterImage]: reference image and monitored image
        """
        monitored_image = GdalRasterImage(mon_file_path)
        reference_image = GdalRasterImage(ref_file_path)

        if not monitored_image.is_compatible_with(reference_image):
            raise KariosException(
                f"""Monitored image geo info not compatible with reference image:
            * Monitored image : {monitored_image.image_information}
            * Reference image : {reference_image.image_information}
            """
            )
        return reference_image, monitored_image

    def _get_mask(self, reference_image: GdalRasterImage) -> GdalRasterImage | None:
        """Get mask as GdalRasterImage and check its compatibility with reference image.
        Notice that mask is in this object config.

        Args:
            reference_image (GdalRasterImage): reference image

        Raises:
            KariosException: if mask is not compatible with reference image

        Returns:
            GdalRasterImage | None: Mask if present in conf and compatible with reference image.
        """

        if self._conf.values.mask_file_path:
            mask = GdalRasterImage(self._conf.values.mask_file_path)
            if not mask.is_compatible_with(reference_image):
                raise KariosException(
                    f"""Mask geo info not compatible with reference image:
                * Mask image : {mask.image_information}
                * Reference image : {reference_image.image_information}
                """
                )
        else:
            mask = None

        return mask

    def _preprocess_large_offset(
        self, reference_image: GdalRasterImage, monitored_image: GdalRasterImage
    ) -> ShiftedImage | None:
        """Computes large offset. Uses `LargeOffsetMatcher`
        to get offsets between reference and monitored image.
        Applies shift on monitored image on X and Y axes independently of each other
        if their offset is higher than a threshold.

        Args:
            reference_image (GdalRasterImage): the reference image to compute offset
            monitored_image (GdalRasterImage): the image to compute the offset

        Returns:
            ShiftedImage | None: the shifted monitored image if shift applied, otherwise None.
        """
        # Compute large offset
        large_offset_matcher = LargeOffsetMatcher(reference_image, monitored_image)
        offsets = large_offset_matcher.match()
        logging.info("Large offset found: %s", offsets)

        offset_threshold = (
            self._conf.shift_image_processing_configuration.bias_correction_min_threshold
        )
        if abs(offsets[1]) < offset_threshold:
            offsets[1] = 0  # do not shift on this axis
        else:
            logger.info("Will apply X offset %s", offsets[1])

        if abs(offsets[0]) < offset_threshold:
            offsets[0] = 0  # do not shift on this axis
        else:
            logger.info("Will apply Y offset %s", offsets[0])

        # apply shift if needed
        if offsets[0] != offsets[1] != 0:
            logger.info("Do shift for large offset")
            new_monitored_array = shift_image(
                monitored_image.array, x_off=offsets[1], y_off=offsets[0]
            )
            mon_filename = os.path.basename(monitored_image.file_name)
            extension = os.path.splitext(mon_filename)[1]
            shifted_image = f"{os.path.splitext(mon_filename)[0]}_shifted"

            shifted_image_path = os.path.join(
                self._conf.values.output_directory, "".join([shifted_image, extension])
            )
            monitored_image.to_raster(shifted_image_path, new_monitored_array)
            shifted_monitored_image = GdalRasterImage(shifted_image_path)

            return ShiftedImage(shifted_monitored_image, offsets[1], offsets[0])

        return None

    def process(self, mon_file_path: str, ref_file_path: str, resume: bool):
        """Orchestrates job to do.
        Process to matching, create plot and csv stat file.

        Args:
            mon_file_path (str): path to image to monitor
            ref_file_path (str): path to reference image used to monitor
            resume (bool): Resume or not previous process. if 'True', then KLT is not run
        """
        logger.info("Process %s", mon_file_path)

        # Prepare output dir
        self._check_output_dir()

        # Prepare inputs
        reference_image, monitored_image = self._get_images(ref_file_path, mon_file_path)
        mask = self._get_mask(reference_image)

        if self._conf.values.enable_large_shift_detection:
            logger.warning("Large shift detection enable, this is an experimental feature.")
            shifted_image = self._preprocess_large_offset(reference_image, monitored_image)

            if shifted_image:
                monitored_image = shifted_image.image
                logger.info("Switch monitored image to %s", monitored_image.filepath)

            points = self._get_points(resume, monitored_image, reference_image, mask)
            if shifted_image:
                logger.info("Apply offset to KLT matcher result")
                points["dx"] = points["dx"] + shifted_image.x_offset
                points["dy"] = points["dy"] + shifted_image.y_offset
        else:
            points = self._get_points(resume, monitored_image, reference_image, mask)

        product_generator = ProductGenerator(self._conf.values, points, reference_image)
        product_generator.generate_products()

        self._plot_overview(monitored_image, reference_image, points)
        self._plot_mean_shift_by_row_col_group(monitored_image, reference_image, points)

        dem = self._get_dem(reference_image)
        if dem:
            self._plot_dem(dem, points, monitored_image, reference_image)
        else:
            logger.info("No DEM file provided, will not plot deviation regarding DEM")

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
    arg_parser = KariosArgumentParser()
    args = arg_parser.parse_args(argv)
    configure_logging(args.debug, not args.no_log_file, args.log_file_path)

    arg_parser.verify_arguments()

    logger.info("Start KARIOS %s with Python %s", __version__, sys.version)
    # set up configuration :
    conf = Configuration(args)
    # do the job
    match_and_plot = MatchAndPlot(conf)
    match_and_plot.process(args.mon, args.ref, args.resume)

    shutil.copy(conf.values.configuration, conf.values.output_directory)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
