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


"""
Represents the configuration of the application.

Contains inputs, outputs and processings parameters.
"""
import json
import logging
import os
from dataclasses import dataclass

from core.errors import ConfigurationError
from core.utils import get_filename

LOGGER = logging.getLogger()


@dataclass
class KLTConfiguration:
    # disable lint for name in order to keep compatible with existing config
    # pylint: disable=invalid-name, too-many-instance-attributes
    """KLT config object"""
    minDistance: int
    blocksize: int
    maxCorners: int
    matching_winsize: int
    qualityLevel: float
    xStart: int
    tile_size: int
    laplacian_kernel_size: int
    outliers_filtering: bool


@dataclass
class OverviewPlotConfiguration:
    """Overview Plot module configuration class"""

    fig_size: int
    shift_colormap: str
    shift_auto_axes_limit: bool
    shift_axes_limit: float
    theta_colormap: str


@dataclass
class ShiftPlotConfiguration:
    """Shift Plot module configuration class"""

    fig_size: int
    scatter_colormap: str
    scatter_auto_limit: bool
    scatter_min_limit: float
    scatter_max_limit: float
    histo_mean_bin_size: int


@dataclass
class CEPlotConfiguration:
    """CE Plot module configuration class"""

    fig_size: int
    ce_scatter_colormap: str


@dataclass
class AccuracyAnalysisConfiguration:
    """Accuracy analysis module configuration class"""

    confidence_threshold: float


@dataclass
class GlobalConfiguration:
    """Global configuration"""

    output_directory: str
    working_image: str
    reference_image: str
    mask: str
    configuration: str
    pixel_size: float
    title_prefix: str
    gen_kp_mask: bool
    gen_delta_raster: bool


class Configuration:
    """Application configuration."""

    def __init__(self, arguments):
        """
        Initialize class and check for configuration files existence.

        Load the configuration as a dictionary if it exists.
            :param self: Instance of the class
            :param arguments: Arguments parsed from the command line
            :param output_path: Where to store outputs
        """
        self.values: GlobalConfiguration = GlobalConfiguration(
            os.path.join(
                arguments.out,
                f"{get_filename(arguments.mon)}_{get_filename(arguments.ref)}",
            ),
            arguments.mon,
            arguments.ref,
            arguments.mask,
            arguments.conf,
            arguments.pixel_size,
            arguments.title_prefix,
            arguments.gen_kp_mask,
            arguments.gen_delta_raster,
        )

        # Read configuration file :
        file_content = self._load_configuration_file(arguments.conf)

        self._load_configuration(file_content["processing_configuration"])

        # TODO: Eventualy add these functions if required
        # self.save_configuration()
        # self.configure_readers()

    def _load_configuration(self, proc_config):
        """Retrieve parameters as dict.

        Args:
          file_content:

        """
        self.klt_configuration = KLTConfiguration(**proc_config["klt_matching"])
        self.accuracy_analysis_configuration = AccuracyAnalysisConfiguration(
            **proc_config["accuracy_analysis"]
        )
        self.overview_plot_configuration = OverviewPlotConfiguration(
            **proc_config["plot_configuration"]["overview"]
        )
        self.shift_plot_configuration = ShiftPlotConfiguration(
            **proc_config["plot_configuration"]["shift"]
        )
        self.ce_plot_configuration = CEPlotConfiguration(**proc_config["plot_configuration"]["ce"])

    def _load_configuration_file(self, filepath):
        """Check that the provided configuration file exists and is valid.
        And load configuration (json)

        Args:
          filepath: The path of the configuration file.

        Returns:

        """
        if os.path.exists(filepath):
            LOGGER.info("** Checking %s", filepath)
            try:
                with open(filepath, encoding="utf-8") as json_file:
                    file_content = json.load(json_file)
            except json.JSONDecodeError as error:
                raise ConfigurationError(
                    f"{filepath} is not a valid configuration file: {error}"
                ) from error

        else:
            LOGGER.error("%s does not exist.", filepath)
            raise ConfigurationError(f"{filepath} does not exist.")
        return file_content

        # self.save_configuration()
        # self.configure_readers()
