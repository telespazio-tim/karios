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
"""KARIOS command line interface module.

Provides command line interface for KARIOS functionality.
"""

import logging
import os
import shutil
import sys
from pathlib import Path, PurePath
from typing import Optional

import rich_click as click
from osgeo import gdal

from karios.api import KariosAPI, RuntimeConfiguration
from karios.core.configuration import ProcessingConfiguration
from karios.log import configure_logging
from karios.version import __version__

logger = logging.getLogger(__name__)


ROOT_DIR = Path(os.path.dirname(__file__)).parent.absolute()

gdal.UseExceptions()

click.rich_click.SHOW_ARGUMENTS = True

# Configure option groups
click.rich_click.OPTION_GROUPS = {
    "karios process": [
        {
            "name": "Processing Options",
            "options": ["--conf", "--resume", "--input-pixel-size"],
        },
        {
            "name": "Output Options",
            "options": [
                "--out",
                "--title-prefix",
                "--generate-key-points-mask",
                "--generate-intermediate-product",
                "--dem-description",
            ],
        },
        {
            "name": "Advanced Options",
            "options": ["--enable-large-shift-detection"],
        },
        {
            "name": "Logging Options",
            "options": ["--debug", "--no-log-file", "--log-file-path"],
        },
    ]
}


def mask_callback(ctx: click.Context, param: click.Parameter, value: Path) -> Path | None:
    """
    Validate if mask value is - or exists.
    It repace click.Path.exists=True as we need to handle spacial case of "-"
    It raises an error with message similar to what click would print if the provided value does not exists.

    Args:
        ctx (click.Context): Click context object containing command state.
        param (click.Parameter): Click parameter that triggered this callback.
        value (Path): The specified mask path.

    Returns:
        Path|None: provided path if exists, None if "-".

    Raises:
        click.BadParameter: If not "-" and the file does not exists.
    """

    # Handle special case for skipped mask
    if param.name == "mask_file":
        if value.exists():
            return value
        if value.name == "-":
            return None

        raise click.BadParameter("File '-' does not exist", ctx, param)


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """KARIOS - KLT Algorithm for Registration of Images from Observing Systems.

    A tool for comparing and matching images using KLT feature tracking.
    """


@cli.command(
    short_help="Performs image matching to detect pixel-level shifts between a monitored image and a reference image"
)
@click.argument(
    "monitored_image", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "reference_image", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "mask_file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    callback=mask_callback,  # handle file exists in place of click for special case -
)
@click.argument(
    "dem_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--conf",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=PurePath(ROOT_DIR, "configuration/processing_configuration.json"),
    help="Configuration file path. Default is the built-in configuration.",
    show_default=True,
)
@click.option(
    "--resume",
    is_flag=True,
    help="Do not run KLT matcher, only accuracy analysis and report generation",
)
@click.option(
    "--input-pixel-size",
    "-pxs",
    type=float,
    default=None,
    help="Input image pixel size in meter. Ignored if image resolution can be read from input image",
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default="results",
    help="Output results folder path",
    show_default=True,
)
@click.option(
    "--title-prefix",
    "-tp",
    type=str,
    default=None,
    help="Add prefix to title of generated output charts (limited to 26 characters)",
    show_default=True,
)
@click.option(
    "--generate-key-points-mask",
    "-kpm",
    is_flag=True,
    help="Generate a tiff mask based on KP from KTL",
)
@click.option(
    "--generate-intermediate-product",
    "-gip",
    is_flag=True,
    help="Generate a two band tiff based on KP with band 1 dx and band 2 dy",
)
@click.option(
    "--dem-description",
    type=str,
    default=None,
    help='DEM source name. Added in generated DEM plots (example: "COPERNICUS DEM resample to 10m")',
    show_default=True,
)
@click.option(
    "--enable-large-shift-detection",
    is_flag=True,
    help="Enable detection and correction of large pixel shifts",
)
@click.option("--debug", "-d", is_flag=True, help="Enable Debug mode")
@click.option("--no-log-file", is_flag=True, help="Do not log in file")
@click.option(
    "--log-file-path",
    type=click.Path(),
    default="karios.log",
    help="Log file path",
    show_default=True,
)
def process(
    monitored_image: Path,
    reference_image: Path,
    mask_file: Optional[Path],
    dem_file: Optional[Path],
    conf: Optional[Path],
    out: Path,
    resume: bool,
    input_pixel_size: Optional[float],
    generate_key_points_mask: bool,
    generate_intermediate_product: bool,
    title_prefix: Optional[str],
    dem_description: Optional[str],
    enable_large_shift_detection: bool,
    no_log_file: bool,
    debug: bool,
    log_file_path: str,
) -> None:
    """\b
    Performs image matching, accuracy analysis and report generation using
    KLT (Kanade-Lucas-Tomasi) feature tracking to detect pixel-level shifts
    between a monitored image and a reference image, where:
    - MONITORED_IMAGE  Path to the image to analyze for shifts/changes
    - REFERENCE_IMAGE  Path to the stable reference image for comparison
    - [MASK_FILE]      Optional mask file to exclude pixels from matching (use '-' to skip when providing only DEM_FILE)
    - [DEM_FILE]       Optional DEM file for altitude-based analysis

    \b

    Examples:

        \b
        # Basic processing

        karios process monitored.tif reference.tif

        \b
        # With mask

        karios process monitored.tif reference.tif mask.tif

        \b
        # With mask and DEM

        karios process monitored.tif reference.tif mask.tif dem.tif --dem-description "SRTM 30m"

        \b
        # DEM only (no mask)

        karios process monitored.tif reference.tif - dem.tif
    """
    # Configure logging
    configure_logging(debug, not no_log_file, log_file_path)

    logger.info("Start KARIOS %s", __version__)

    try:

        # Load configuration from file
        logger.info("Load config from file %s", conf)
        processing_configuration = ProcessingConfiguration.from_file(conf)

        # Create output directory
        output_dir = out / f"{monitored_image.stem}_{reference_image.stem}"
        os.makedirs(output_dir, exist_ok=True)

        # Create runtime configuration
        runtime_configuration = RuntimeConfiguration(
            output_directory=output_dir,
            pixel_size=input_pixel_size,
            title_prefix=title_prefix,
            gen_kp_mask=generate_key_points_mask,
            gen_delta_raster=generate_intermediate_product,
            dem_description=dem_description,
            enable_large_shift_detection=enable_large_shift_detection,
        )

        # Validate configuration
        _validate_configuration(runtime_configuration, dem_file)

        # Initialize API
        api = KariosAPI(processing_configuration, runtime_configuration)

        # Run processing pipeline with input files as parameters
        match_result, accuracy, reports = api.process(
            monitored_image, reference_image, mask_file, dem_file, resume
        )

        # Copy configuration to output directory
        shutil.copy(conf, output_dir)

        logger.info("Processing completed successfully")
        logger.info("Results written to %s", output_dir)

        _print_summary(match_result, accuracy, reports)

        return 0

    except Exception as e:
        logger.error("Error during processing: %s", str(e), exc_info=debug)
        return 1


def _validate_configuration(config: RuntimeConfiguration, dem_file: Optional[Path]) -> None:
    """Validate configuration parameters and input files.

    Args:
        config: RuntimeConfiguration object to validate
        dem_file: Optional DEM file path

    Raises:
        ValueError: If configuration is invalid
    """

    # Check if DEM description is provided without DEM file
    if not dem_file and config.dem_description:
        logger.warning("DEM description provided but no DEM file, DEM description will be ignored")

    # Check if DEM file is provided without description
    if dem_file and not config.dem_description:
        logger.warning("DEM provided but no DEM description, consider adding a description")

    # Validate title prefix length
    if config.title_prefix and len(config.title_prefix) > 26:
        raise ValueError("Title prefix is too long (>26 characters)")


def _print_summary(match_result, accuracy, reports) -> None:
    """Print summary of processing results.

    Args:
        match_result: Match result object
        accuracy: Accuracy analysis object
        reports: Report paths object
    """
    click.echo("\nKARIOS Processing Summary")
    click.echo("=======================")

    click.echo(f"\nMatched {len(match_result.points)} keypoints")
    click.echo(
        f"Valid pixels: {accuracy.valid_pixels}/{accuracy.total_pixels} "
        + f"({accuracy.valid_pixels/accuracy.total_pixels*100:.2f}%)"
    )

    click.echo("\nAccuracy Statistics:")
    click.echo(f"  Mean X: {accuracy.mean_x:.3f} pixels")
    click.echo(f"  Mean Y: {accuracy.mean_y:.3f} pixels")
    click.echo(f"  Standard Deviation X: {accuracy.std_x:.3f} pixels")
    click.echo(f"  Standard Deviation Y: {accuracy.std_y:.3f} pixels")
    click.echo(f"  CE90: {accuracy.ce90:.3f}")
    click.echo(f"  CE95: {accuracy.ce95:.3f}")

    click.echo("\nGenerated outputs:")
    click.echo(f"  Overview plot: {reports.overview_plot}")
    click.echo(f"  DX plot: {reports.dx_plot}")
    click.echo(f"  DY plot: {reports.dy_plot}")
    click.echo(f"  CE plot: {reports.ce_plot}")

    if reports.dem_plots:
        click.echo(f"  DEM plots: {len(reports.dem_plots)}")
        for plot in reports.dem_plots:
            click.echo(f"    - {plot}")

    if reports.products:
        click.echo(f"  Products: {len(reports.products)}")
        for product in reports.products:
            click.echo(f"    - {product}")


if __name__ == "__main__":
    sys.exit(cli())
