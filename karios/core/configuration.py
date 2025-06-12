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

"""
Represents the configuration of the application.

Contains inputs, outputs and processings parameters.
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from karios.core.errors import ConfigurationError

LOGGER = logging.getLogger(__name__)


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
    """Shift by Row Col Plot module configuration class"""

    fig_size: int
    scatter_colormap: str
    scatter_auto_limit: bool
    scatter_min_limit: float
    scatter_max_limit: float
    histo_mean_bin_size: int


@dataclass
class DemPlotConfiguration:
    """Shift by DEM Plot module configuration class"""

    fig_size: int
    show_fliers: bool
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
class ShiftConfiguration:
    """Large shift image preprocessing configuration"""

    bias_correction_min_threshold: int


class ProcessingConfiguration:
    """Application configuration."""

    def __init__(self):
        """Initialize Configuration class."""
        self.klt_configuration: Optional[KLTConfiguration] = None
        self.shift_image_processing_configuration: Optional[ShiftConfiguration] = None
        self.accuracy_analysis_configuration: Optional[AccuracyAnalysisConfiguration] = None
        self.overview_plot_configuration: Optional[OverviewPlotConfiguration] = None
        self.shift_plot_configuration: Optional[ShiftPlotConfiguration] = None
        self.dem_plot_configuration: Optional[DemPlotConfiguration] = None
        self.ce_plot_configuration: Optional[CEPlotConfiguration] = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ProcessingConfiguration":
        """Create configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            ProcessingConfiguration: Configured instance
        """
        instance = cls()
        instance._load_configuration(config_dict)
        return instance

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "ProcessingConfiguration":
        """Load configuration from a file.

        Args:
            filepath: Path to configuration file

        Returns:
            ProcessingConfiguration: Configured instance
        """
        filepath_str = str(filepath)
        if not os.path.exists(filepath_str):
            LOGGER.error("%s does not exist.", filepath_str)
            raise ConfigurationError(f"{filepath_str} does not exist.")

        try:
            with open(filepath_str, encoding="utf-8") as json_file:
                config_dict = json.load(json_file)
        except json.JSONDecodeError as error:
            raise ConfigurationError(
                f"{filepath_str} is not a valid configuration file: {error}"
            ) from error

        return cls.from_dict(config_dict)

    def _load_configuration(self, config_dict: dict[str, Any]) -> None:
        """Load configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration
        """
        self.klt_configuration = KLTConfiguration(
            **config_dict["processing_configuration"]["klt_matching"]
        )
        self.shift_image_processing_configuration = ShiftConfiguration(
            **config_dict["processing_configuration"]["shift_image_processing"]
        )
        self.accuracy_analysis_configuration = AccuracyAnalysisConfiguration(
            **config_dict["processing_configuration"]["accuracy_analysis"]
        )
        self.overview_plot_configuration = OverviewPlotConfiguration(
            **config_dict["plot_configuration"]["overview"]
        )
        self.shift_plot_configuration = ShiftPlotConfiguration(
            **config_dict["plot_configuration"]["shift"]
        )
        self.dem_plot_configuration = DemPlotConfiguration(
            **config_dict["plot_configuration"]["dem"]
        )
        self.ce_plot_configuration = CEPlotConfiguration(**config_dict["plot_configuration"]["ce"])

    def _load_configuration_file(self, filepath: str) -> dict[str, Any]:
        """Check that the provided configuration file exists and is valid.
        And load configuration (json)

        Args:
            filepath: The path of the configuration file.

        Returns:
            dict[str, Any]: Loaded configuration dictionary

        Raises:
            ConfigurationError: If file doesn't exist or contains invalid JSON
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
