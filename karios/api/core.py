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
"""KARIOS API core module.

Provides the main entry point for the KARIOS API functionality.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from karios.accuracy_analysis.accuracy_statistics import GeometricStat
from karios.api.config import RuntimeConfiguration
from karios.core.configuration import ProcessingConfiguration
from karios.core.errors import KariosException
from karios.core.image import GdalRasterImage, get_image_resolution, shift_image
from karios.core.utils import get_filename
from karios.matcher.klt import KLT
from karios.matcher.large_offset import LargeOffsetMatcher
from karios.report.circular_error_plot import CircularErrorPlot
from karios.report.overview_plot import OverviewPlot
from karios.report.product_generator import ProductGenerator
from karios.report.shift_by_alt_plot import MeanShiftByAltitudeGroupPlot
from karios.report.shift_by_row_col_plot import MeanShiftByRowColGroupPlot

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of image matching process."""

    points: pd.DataFrame
    reference_image: GdalRasterImage
    monitored_image: GdalRasterImage
    mask: Optional[GdalRasterImage] = None


@dataclass
class ShiftedImage:
    """Describe a shifted image"""

    image: GdalRasterImage
    """shifted image"""
    x_offset: int
    """X/col offset compared to original image in pixel"""
    y_offset: int
    """Y/row offset compared to original image in pixel"""


@dataclass
class AccuracyAnalysis:
    """Result of accuracy analysis."""

    statistics: GeometricStat
    mean_x: float
    mean_y: float
    std_x: float
    std_y: float
    ce90: float
    ce95: float
    valid_pixels: int
    total_pixels: int


@dataclass
class ReportPaths:
    """Paths to generated reports."""

    overview_plot: str
    dx_plot: str
    dy_plot: str
    ce_plot: str
    dem_plots: list[str]
    products: list[str]


class KariosAPI:
    """Main API class for KARIOS functionality.

    This class provides the primary interface for image matching, accuracy analysis,
    and report generation. It separates processing configuration from input data,
    allowing the same configuration to be reused for multiple image pairs.

    Example:
        >>> # Create configuration once
        >>> config = RuntimeConfiguration(
        ...     output_directory="/results",
        ...     gen_kp_mask=True,
        ...     pixel_size=1.0
        ... )
        >>> api = KariosAPI(processing_config, config)
        >>>
        >>> # Process multiple image pairs with same configuration
        >>> result1 = api.process("image1.tif", "reference.tif")
        >>> result2 = api.process("image2.tif", "reference.tif")
    """

    def __init__(
        self,
        processing_configuration: ProcessingConfiguration,
        runtime_configuration: RuntimeConfiguration,
    ):
        """Initialize the KARIOS API with processing and runtime configuration.

        Args:
            processing_configuration: Configuration for KLT matching, accuracy analysis,
                and plotting parameters
            runtime_configuration: Runtime configuration specifying output settings,
                optional mask/DEM files, and processing flags.
                Does not include input image paths - these are
                provided to processing methods.
        """
        self._processing_configuration = processing_configuration
        self._runtime_configuration = runtime_configuration

        # Initialize KLT matcher
        self._klt = KLT(
            self._processing_configuration.klt_configuration,
            self._runtime_configuration.gen_delta_raster,
            self._runtime_configuration.output_directory,
        )

        # Prepare output dir
        self._check_output_dir()

    def match_images(
        self,
        monitored_image_path: Path,
        reference_image_path: Path,
        mask_file_path: Optional[Path] = None,
        resume: bool = False,
    ) -> MatchResult:
        """Match the monitored image against the reference image.

        Args:
            monitored_image_path: Path to the monitored image
            reference_image_path: Path to the reference image
            mask_file_path: Optional path to mask file for excluding pixels from matching.
                Mask should be compatible with the monitored image.
            resume: Whether to resume from previous analysis

        Returns:
            MatchResult: Object containing match points and statistics
        """

        # Load images
        reference_image, monitored_image = self._load_images(
            reference_image_path, monitored_image_path
        )

        # Load mask if provided
        mask = self._load_mask(monitored_image, mask_file_path)

        # Handle large offset detection if enabled
        if self._runtime_configuration.enable_large_shift_detection:
            logger.warning("Large shift detection enable, this is an experimental feature.")
            shifted_image = self._detect_large_offset(reference_image, monitored_image)

            if shifted_image:
                monitored_image.clear_cache()  # force clean
                monitored_image = shifted_image.image
                logger.info("Switch monitored image to %s", monitored_image.filepath)
                points = self._get_match_points(resume, monitored_image, reference_image, mask)

                # Apply offset to match results
                logger.info("Apply offset to KLT matcher result")
                points["dx"] = points["dx"] + shifted_image.x_offset
                points["dy"] = points["dy"] + shifted_image.y_offset
            else:
                logger.info("Large shift too tight to be applyed, do not apply")
                points = self._get_match_points(resume, monitored_image, reference_image, mask)
        else:
            points = self._get_match_points(resume, monitored_image, reference_image, mask)

        return MatchResult(
            points=points,
            reference_image=reference_image,
            monitored_image=monitored_image,
            mask=mask,
        )

    def analyze_accuracy(self, match_result: MatchResult) -> AccuracyAnalysis:
        """Analyze the accuracy of image matching.

        Args:
            match_result: Result from match_images

        Returns:
            AccuracyAnalysis: Object containing accuracy statistics
        """
        acc_config = self._processing_configuration.accuracy_analysis_configuration
        stats = GeometricStat(
            acc_config,
            match_result.points,
            match_result.monitored_image.have_pixel_resolution(),
        )

        # Compute number of valid pixels considering mask if any
        masked_image = match_result.monitored_image.array
        if match_result.mask is not None:
            masked_image = np.copy(match_result.monitored_image.array)
            masked_image[match_result.mask.array == 0] = 0

        nb_valid_pixel = np.count_nonzero(masked_image)
        logger.info("NB of valid px %s", nb_valid_pixel)
        logger.info(
            "NB of total px %s",
            match_result.monitored_image.x_size * match_result.monitored_image.y_size,
        )
        total_pixels = match_result.monitored_image.x_size * match_result.monitored_image.y_size

        # Compute stats
        stats.compute_stats(nb_valid_pixel)

        # Save statistics file
        stats.update_statistic_file(
            match_result.reference_image.file_name,
            match_result.monitored_image.file_name,
            str(Path(self._runtime_configuration.output_directory) / "correl_res.txt"),
        )

        # Calculate CE metrics
        img_res = get_image_resolution(
            match_result.monitored_image,
            match_result.reference_image,
            self._runtime_configuration.pixel_size,
        )
        img_res = img_res if img_res is not None else 1.0
        ce90 = stats.compute_percentile(0.9, img_res)
        ce95 = stats.compute_percentile(0.95, img_res)

        return AccuracyAnalysis(
            statistics=stats,
            mean_x=stats.mean_x,
            mean_y=stats.mean_y,
            std_x=stats.std_x,
            std_y=stats.std_y,
            ce90=ce90,
            ce95=ce95,
            valid_pixels=nb_valid_pixel,
            total_pixels=total_pixels,
        )

    def generate_reports(
        self,
        match_result: MatchResult,
        accuracy_analysis: AccuracyAnalysis,
        dem_file_path: Optional[Path] = None,
    ) -> ReportPaths:
        """Generate reports and visualizations.

        Args:
            match_result: Result from match_images
            accuracy_analysis: Result from analyze_accuracy
            dem_file_path: Optional path to DEM file for altitude-based analysis.
                DEM should be compatible with the reference image.

        Returns:
            ReportPaths: Object containing paths to generated reports
        """

        output_dir = self._runtime_configuration.output_directory

        # Generate products (mask, rasters, etc.)
        product_generator = ProductGenerator(
            self._runtime_configuration,
            match_result.points,
            match_result.reference_image,
        )
        product_generator.generate_products()

        # Generate standard plots...
        overview_path = self._generate_overview_plot(match_result, output_dir)
        dx_plot_path = self._generate_dx_plot(match_result, output_dir)
        dy_plot_path = self._generate_dy_plot(match_result, output_dir)
        ce_plot_path = self._generate_ce_plot(match_result, accuracy_analysis, output_dir)

        # Generate DEM plots if DEM is provided
        dem_plots = self._generate_dem_plots(match_result, output_dir, dem_file_path)

        # make sure all are closed
        plt.close("all")

        # List product paths
        product_paths = []
        if self._runtime_configuration.gen_kp_mask:
            product_paths.append(str(output_dir / "kp_mask.tif"))
        if self._runtime_configuration.gen_delta_raster:
            product_paths.append(str(output_dir / "kp_delta.tif"))
        if match_result.reference_image.get_epsg():
            product_paths.append(str(output_dir / "kp_delta.json"))

        return ReportPaths(
            overview_plot=str(overview_path),
            dx_plot=str(dx_plot_path),
            dy_plot=str(dy_plot_path),
            ce_plot=str(ce_plot_path),
            dem_plots=dem_plots,
            products=product_paths,
        )

    def process(
        self,
        monitored_image_path: Path,
        reference_image_path: Path,
        mask_file_path: Optional[Path] = None,
        dem_file_path: Optional[Path] = None,
        resume: bool = False,
    ) -> tuple[MatchResult, AccuracyAnalysis, ReportPaths]:
        """Complete processing pipeline combining matching, analysis and reporting.

        This method performs the full KARIOS workflow:
        1. Image matching using KLT feature tracking
        2. Accuracy analysis with statistical metrics
        3. Report and visualization generation

        The processing configuration and output settings are defined by the
        RuntimeConfiguration provided during API initialization, while the specific
        input files are provided as parameters to this method.

        Args:
            monitored_image_path: Path to the image to be analyzed for shifts/changes
            reference_image_path: Path to the stable reference image for comparison
            mask_file_path: Optional Path to mask file for excluding pixels from matching.
                        Mask should be compatible with the monitored image.
            dem_file_path: Optional Path to DEM file for altitude-based analysis.
                        DEM should be compatible with the reference image.
            resume: Whether to resume from previous analysis (skip KLT if CSV exists)

        Returns:
            Tuple containing:
            - MatchResult: Key point matches and image objects
            - AccuracyAnalysis: Statistical accuracy metrics (CE90, RMSE, etc.)
            - ReportPaths: File paths to generated reports and visualizations

        Raises:
            KariosException: If images are not compatible (different projections/sizes)

        Example:
            >>> api = KariosAPI(proc_config, runtime_config)
            >>> match, accuracy, reports = api.process("mon.tif", "ref.tif", "mask.tif", "dem.tif")
            >>> print(f"CE90: {accuracy.ce90:.3f}")
        """

        logger.info("Process %s", monitored_image_path)

        match_result = self.match_images(
            monitored_image_path, reference_image_path, mask_file_path, resume
        )

        accuracy = self.analyze_accuracy(match_result)

        reports = self.generate_reports(match_result, accuracy, dem_file_path)

        return match_result, accuracy, reports

    def _check_output_dir(self):
        output_dir_path = Path(self._runtime_configuration.output_directory)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True)
        else:
            logger.warning(
                "Output dir %s already exists, some files could be overridden.",
                output_dir_path,
            )

    def _check_quality(self, monitored_image: GdalRasterImage, reference_image: GdalRasterImage):
        """This function verify input image quality by:
        - verifying dynamic range at 2 and 98 percentile

        Args:
            monitored_image (GdalRasterImage): monitored image to check
            reference_image (GdalRasterImage): reference image to check
        """

        min_max = np.nanpercentile(monitored_image.array, [2, 98])
        if min_max[1] - min_max[0] <= 10:
            logger.warning("Low dynamic range detected for monitored, you could get poor results")

        min_max = np.nanpercentile(reference_image.array, [2, 98])
        if min_max[1] - min_max[0] <= 10:
            logger.warning("Low dynamic range detected for reference, you could get poor results")

    def _load_images(
        self, ref_file_path: Path, mon_file_path: Path
    ) -> tuple[GdalRasterImage, GdalRasterImage]:
        """Build GdalRasterImage objects and check compatibility.

        Args:
            ref_file_path: File path to reference image
            mon_file_path: File path to monitored image

        Raises:
            KariosException: If monitored image is not compatible with reference image

        Returns:
            Tuple containing reference image and monitored image
        """
        monitored_image = GdalRasterImage(mon_file_path)
        reference_image = GdalRasterImage(ref_file_path)

        self._check_quality(monitored_image, reference_image)

        if not monitored_image.is_compatible_with(reference_image):
            raise KariosException(
                f"""Monitored image geo info not compatible with reference image:
            * Monitored image : {monitored_image.image_information}
            * Reference image : {reference_image.image_information}
            """
            )

        return reference_image, monitored_image

    def _load_mask(
        self, monitored_image: GdalRasterImage, mask_file_path: Optional[Path]
    ) -> Optional[GdalRasterImage]:
        """Load mask as GdalRasterImage and check compatibility with monitored image.

        Args:
            monitored_image: Monitored image to check compatibility against
            mask_file_path: Path to mask file

        Raises:
            KariosException: If mask is not compatible with monitored image

        Returns:
            Mask if compatible with monitored image, None if no mask provided
        """
        if not mask_file_path:
            logger.info("No mask provided")
            return None

        logger.info("Load mask file %s", mask_file_path)
        mask = GdalRasterImage(mask_file_path)
        if not mask.is_compatible_with(monitored_image):
            raise KariosException(
                f"""Mask geo info not compatible with monitored image:
            * Mask image : {mask.image_information}
            * Monitored image : {monitored_image.image_information}
            """
            )
        return mask

    def _load_dem(
        self, reference_image: GdalRasterImage, dem_file_path: Optional[Path]
    ) -> Optional[GdalRasterImage]:
        """Load DEM as GdalRasterImage and check compatibility with reference image.

        Args:
            reference_image: Reference image to check compatibility against
            dem_file_path: Path to DEM file

        Raises:
            KariosException: If DEM is not compatible with reference image

        Returns:
            DEM if present and compatible with reference image, None if no DEM provided
        """
        if not dem_file_path:
            return None

        dem = GdalRasterImage(dem_file_path)
        if not dem.is_compatible_with(reference_image):
            raise KariosException(
                f"""DEM geo info not compatible with reference image shape or resolution:
            * DEM image : {dem.image_information}
            * Reference image : {reference_image.image_information}
            """
            )
        return dem

    def _detect_large_offset(
        self, reference_image: GdalRasterImage, monitored_image: GdalRasterImage
    ) -> Optional[ShiftedImage]:
        """Detect and compensate for large offsets between images.

        Args:
            reference_image: Reference image
            monitored_image: Monitored image

        Returns:
            ShiftedImage if shift was applied, None otherwise
        """
        # Compute large offset
        large_offset_matcher = LargeOffsetMatcher(reference_image, monitored_image)
        offsets = large_offset_matcher.match()
        logging.info("Large offset found: %s", offsets)

        offset_threshold = (
            self._processing_configuration.shift_image_processing_configuration.bias_correction_min_threshold
        )

        # Only apply offset if it exceeds threshold
        if abs(offsets[1]) < offset_threshold:
            offsets[1] = 0  # do not shift on this axis
        else:
            logger.info("Will apply X offset %s", offsets[1])

        if abs(offsets[0]) < offset_threshold:
            offsets[0] = 0  # do not shift on this axis
        else:
            logger.info("Will apply Y offset %s", offsets[0])

        # Apply shift if needed
        if offsets[0] != 0 or offsets[1] != 0:
            logger.info("Do shift for large offset")
            new_monitored_array = shift_image(
                monitored_image.array, x_off=offsets[1], y_off=offsets[0]
            )
            mon_filename = Path(monitored_image.file_name)
            shifted_name = f"{mon_filename.stem}_shifted{mon_filename.suffix}"

            shifted_image_path = Path(self._runtime_configuration.output_directory) / shifted_name
            monitored_image.to_raster(str(shifted_image_path), new_monitored_array)
            shifted_monitored_image = GdalRasterImage(str(shifted_image_path))

            return ShiftedImage(shifted_monitored_image, offsets[1], offsets[0])

        return None

    def _get_match_points(
        self,
        resume: bool,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        mask: Optional[GdalRasterImage],
    ) -> pd.DataFrame:
        """Get match points either by running KLT or reading from CSV.

        Args:
            resume: Whether to resume from previous analysis
            monitored_image: Monitored image
            reference_image: Reference image
            mask: Optional mask image

        Returns:
            DataFrame containing match points
        """

        filename = f"KLT_matcher_{get_filename(monitored_image.filepath)}_{get_filename(reference_image.filepath)}"
        csv_file = Path(self._runtime_configuration.output_directory) / f"{filename}.csv"

        if not resume:
            # Run matcher if not resuming
            if csv_file.exists():
                logger.warning("CSV file exists, will overwrite it: %s", str(csv_file))
                csv_file.unlink()
            points = self._compute_matches(monitored_image, reference_image, mask, csv_file)
        elif not csv_file.exists():
            # Run matcher if resuming but CSV doesn't exist
            logger.warning("Cannot resume, CSV file missing, create it : %s", str(csv_file))
            points = self._compute_matches(monitored_image, reference_image, mask, csv_file)
        else:
            # Load from CSV if resuming and CSV exists
            logger.info("Load CSV : %s", str(csv_file))
            points = pd.read_csv(csv_file, sep=";", index_col=False)

        return points

    def _compute_matches(
        self,
        monitored_image: GdalRasterImage,
        reference_image: GdalRasterImage,
        mask: Optional[GdalRasterImage],
        csv_file: Path,
    ) -> pd.DataFrame:
        """Compute matches using KLT tracker.

        Args:
            monitored_image: Monitored image
            reference_image: Reference image
            mask: Optional mask image
            csv_file: Path to save CSV results

        Returns:
            DataFrame containing match points
        """
        dataframe_gen = self._klt.match(monitored_image, reference_image, mask)
        return self._handle_klt_results(dataframe_gen, csv_file)

    def _handle_klt_results(self, results: pd.DataFrame, csv_file: Path) -> pd.DataFrame:
        """Process KLT results and save to CSV.

        Args:
            results: Generator producing DataFrames with KLT results
            csv_file: Path to save CSV results

        Returns:
            Combined DataFrame with all results
        """
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

    def _generate_overview_plot(self, match_result: MatchResult, output_dir: Path) -> Path:
        """Generate overview plot.

        Args:
            match_result: Match result
            output_dir: Output directory

        Returns:
            Path to the generated plot
        """
        overview_plot = OverviewPlot(
            self._processing_configuration.overview_plot_configuration,
            match_result.monitored_image,
            match_result.reference_image,
            match_result.points,
            self._runtime_configuration.title_prefix,
        )
        overview_path = output_dir / "01_overview.png"
        overview_plot.plot(overview_path)
        return overview_path

    def _generate_dx_plot(self, match_result: MatchResult, output_dir: Path) -> Path:
        """Generate dx shift plot.

        Args:
            match_result: Match result
            output_dir: Output directory

        Returns:
            Path to the generated plot
        """
        dx_plot = MeanShiftByRowColGroupPlot(
            self._processing_configuration.shift_plot_configuration,
            match_result.monitored_image,
            match_result.reference_image,
            match_result.points,
            "dx",
            self._runtime_configuration.title_prefix,
        )
        dx_path = output_dir / "02_dx.png"
        dx_plot.plot(dx_path)
        return dx_path

    def _generate_dy_plot(self, match_result: MatchResult, output_dir: Path) -> Path:
        """Generate dy shift plot.

        Args:
            match_result: Match result
            output_dir: Output directory

        Returns:
            Path to the generated plot
        """
        dy_plot = MeanShiftByRowColGroupPlot(
            self._processing_configuration.shift_plot_configuration,
            match_result.monitored_image,
            match_result.reference_image,
            match_result.points,
            "dy",
            self._runtime_configuration.title_prefix,
        )
        dy_path = output_dir / "03_dy.png"
        dy_plot.plot(dy_path)
        return dy_path

    def _generate_ce_plot(
        self,
        match_result: MatchResult,
        accuracy_analysis: AccuracyAnalysis,
        output_dir: Path,
    ) -> Path:
        """Generate circular error plot.

        Args:
            match_result: Match result
            accuracy_analysis: Accuracy analysis
            output_dir: Output directory

        Returns:
            Path to the generated plot
        """
        ce_path = output_dir / "04_ce.png"

        monitored_image_resolution = get_image_resolution(
            match_result.monitored_image,
            match_result.reference_image,
            self._runtime_configuration.pixel_size,
        )

        ce_plot = CircularErrorPlot(
            self._processing_configuration.ce_plot_configuration,
            match_result.monitored_image,
            match_result.reference_image,
            accuracy_analysis.statistics,
            monitored_image_resolution,
            self._runtime_configuration.title_prefix,
        )
        ce_plot.plot(ce_path)
        return ce_path

    def _generate_dem_plots(
        self, match_result: MatchResult, output_dir: Path, dem_file_path: Optional[Path]
    ) -> list[str]:
        """Generate DEM-based plots if DEM is available.

        Args:
            match_result: Match result
            output_dir: Output directory
            dem_file_path: Optional path to DEM file

        Returns:
            List of paths to generated DEM plots
        """
        dem_paths = []

        # Try to load DEM
        dem = self._load_dem(match_result.reference_image, dem_file_path)
        if not dem:
            if dem_file_path:
                logger.warning("DEM file provided but not compatible with reference image")
            else:
                logger.info("No DEM file provided, will not plot deviation regarding DEM")
            return dem_paths

        # Extract DEM values for each point
        points = match_result.points.copy()
        x_index = points["x0"].to_numpy().astype(int)
        y_index = points["y0"].to_numpy().astype(int)
        points["alt"] = dem.array[y_index, x_index]

        x_index = None
        y_index = None

        # Configure plot parameters
        conf = self._processing_configuration.dem_plot_configuration

        # Compute min and max for consistent axes
        maxi = math.ceil(max(points["dx"].max(), points["dy"].max(), points["radial error"].max()))
        mini = math.floor(min(points["dx"].min(), points["dy"].min(), points["radial error"].min()))

        # Generate plots for each dimension
        for dimension in ["dx", "dy", "radial error"]:
            report = MeanShiftByAltitudeGroupPlot(
                conf,
                match_result.monitored_image,
                match_result.reference_image,
                dem,
                points,
                dimension,
                self._runtime_configuration.title_prefix,
                self._runtime_configuration.dem_description,
                mini,
                maxi,
            )
            poster_path = output_dir / f"dem_{dimension.replace(' ', '_')}.png"
            report.plot(poster_path)
            dem_paths.append(str(poster_path))

        points = None

        return dem_paths
