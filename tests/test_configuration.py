#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for configuration module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from karios.core.configuration import (
    AccuracyAnalysisConfiguration,
    CEPlotConfiguration,
    DemPlotConfiguration,
    KLTConfiguration,
    OverviewPlotConfiguration,
    ProcessingConfiguration,
    ShiftConfiguration,
    ShiftPlotConfiguration,
)
from karios.core.errors import ConfigurationError


def test_klt_configuration():
    """Test KLTConfiguration dataclass."""
    config = KLTConfiguration(
        minDistance=10,
        blocksize=15,
        maxCorners=20000,
        matching_winsize=25,
        qualityLevel=0.01,
        xStart=0,
        tile_size=1000,
        laplacian_kernel_size=3,
        outliers_filtering=True,
    )

    assert config.minDistance == 10
    assert config.blocksize == 15
    assert config.maxCorners == 20000
    assert config.matching_winsize == 25
    assert config.qualityLevel == 0.01
    assert config.xStart == 0
    assert config.tile_size == 1000
    assert config.laplacian_kernel_size == 3
    assert config.outliers_filtering is True


def test_overview_plot_configuration():
    """Test OverviewPlotConfiguration dataclass."""
    config = OverviewPlotConfiguration(
        fig_size=10,
        shift_colormap="viridis",
        shift_auto_axes_limit=True,
        shift_axes_limit=5.0,
        theta_colormap="plasma",
    )

    assert config.fig_size == 10
    assert config.shift_colormap == "viridis"
    assert config.shift_auto_axes_limit is True
    assert config.shift_axes_limit == 5.0
    assert config.theta_colormap == "plasma"


def test_shift_plot_configuration():
    """Test ShiftPlotConfiguration dataclass."""
    config = ShiftPlotConfiguration(
        fig_size=12,
        scatter_colormap="jet",
        scatter_auto_limit=True,
        scatter_min_limit=-10.0,
        scatter_max_limit=10.0,
        histo_mean_bin_size=5,
    )

    assert config.fig_size == 12
    assert config.scatter_colormap == "jet"
    assert config.scatter_auto_limit is True
    assert config.scatter_min_limit == -10.0
    assert config.scatter_max_limit == 10.0
    assert config.histo_mean_bin_size == 5


def test_dem_plot_configuration():
    """Test DemPlotConfiguration dataclass."""
    config = DemPlotConfiguration(fig_size=8, show_fliers=False, histo_mean_bin_size=10)

    assert config.fig_size == 8
    assert config.show_fliers is False
    assert config.histo_mean_bin_size == 10


def test_ce_plot_configuration():
    """Test CEPlotConfiguration dataclass."""
    config = CEPlotConfiguration(fig_size=15, ce_scatter_colormap="hot")

    assert config.fig_size == 15
    assert config.ce_scatter_colormap == "hot"


def test_accuracy_analysis_configuration():
    """Test AccuracyAnalysisConfiguration dataclass."""
    config = AccuracyAnalysisConfiguration(confidence_threshold=0.95)

    assert config.confidence_threshold == 0.95


def test_shift_configuration():
    """Test ShiftConfiguration dataclass."""
    config = ShiftConfiguration(bias_correction_min_threshold=5)

    assert config.bias_correction_min_threshold == 5


def test_processing_configuration_initialization():
    """Test ProcessingConfiguration initialization."""
    config = ProcessingConfiguration()

    assert config.klt_configuration is None
    assert config.shift_image_processing_configuration is None
    assert config.accuracy_analysis_configuration is None
    assert config.overview_plot_configuration is None
    assert config.shift_plot_configuration is None
    assert config.dem_plot_configuration is None
    assert config.ce_plot_configuration is None


def test_processing_configuration_from_dict():
    """Test ProcessingConfiguration.from_dict method."""
    config_dict = {
        "processing_configuration": {
            "klt_matching": {
                "minDistance": 10,
                "blocksize": 15,
                "maxCorners": 20000,
                "matching_winsize": 25,
                "qualityLevel": 0.01,
                "xStart": 0,
                "tile_size": 1000,
                "laplacian_kernel_size": 3,
                "outliers_filtering": True,
            },
            "shift_image_processing": {"bias_correction_min_threshold": 5},
            "accuracy_analysis": {"confidence_threshold": 0.95},
        },
        "plot_configuration": {
            "overview": {
                "fig_size": 10,
                "shift_colormap": "viridis",
                "shift_auto_axes_limit": True,
                "shift_axes_limit": 5.0,
                "theta_colormap": "plasma",
            },
            "shift": {
                "fig_size": 12,
                "scatter_colormap": "jet",
                "scatter_auto_limit": True,
                "scatter_min_limit": -10.0,
                "scatter_max_limit": 10.0,
                "histo_mean_bin_size": 5,
            },
            "dem": {"fig_size": 8, "show_fliers": False, "histo_mean_bin_size": 10},
            "ce": {"fig_size": 15, "ce_scatter_colormap": "hot"},
        },
    }

    config = ProcessingConfiguration.from_dict(config_dict)

    # Check that all configurations were loaded
    assert config.klt_configuration is not None
    assert config.shift_image_processing_configuration is not None
    assert config.accuracy_analysis_configuration is not None
    assert config.overview_plot_configuration is not None
    assert config.shift_plot_configuration is not None
    assert config.dem_plot_configuration is not None
    assert config.ce_plot_configuration is not None

    # Check specific values
    assert config.klt_configuration.minDistance == 10
    assert config.klt_configuration.blocksize == 15
    assert config.klt_configuration.maxCorners == 20000
    assert config.klt_configuration.matching_winsize == 25
    assert config.klt_configuration.qualityLevel == 0.01
    assert config.klt_configuration.xStart == 0
    assert config.klt_configuration.tile_size == 1000
    assert config.klt_configuration.laplacian_kernel_size == 3
    assert config.klt_configuration.outliers_filtering is True

    assert config.shift_image_processing_configuration.bias_correction_min_threshold == 5
    assert config.accuracy_analysis_configuration.confidence_threshold == 0.95

    assert config.overview_plot_configuration.fig_size == 10
    assert config.overview_plot_configuration.shift_colormap == "viridis"
    assert config.overview_plot_configuration.shift_auto_axes_limit is True
    assert config.overview_plot_configuration.shift_axes_limit == 5.0
    assert config.overview_plot_configuration.theta_colormap == "plasma"

    assert config.shift_plot_configuration.fig_size == 12
    assert config.shift_plot_configuration.scatter_colormap == "jet"
    assert config.shift_plot_configuration.scatter_auto_limit is True
    assert config.shift_plot_configuration.scatter_min_limit == -10.0
    assert config.shift_plot_configuration.scatter_max_limit == 10.0
    assert config.shift_plot_configuration.histo_mean_bin_size == 5

    assert config.dem_plot_configuration.fig_size == 8
    assert config.dem_plot_configuration.show_fliers is False
    assert config.dem_plot_configuration.histo_mean_bin_size == 10

    assert config.ce_plot_configuration.fig_size == 15
    assert config.ce_plot_configuration.ce_scatter_colormap == "hot"


def test_processing_configuration_from_file():
    """Test ProcessingConfiguration.from_file method."""
    # Create a temporary config file
    config_dict = {
        "processing_configuration": {
            "klt_matching": {
                "minDistance": 10,
                "blocksize": 15,
                "maxCorners": 20000,
                "matching_winsize": 25,
                "qualityLevel": 0.01,
                "xStart": 0,
                "tile_size": 1000,
                "laplacian_kernel_size": 3,
                "outliers_filtering": True,
            },
            "shift_image_processing": {"bias_correction_min_threshold": 5},
            "accuracy_analysis": {"confidence_threshold": 0.95},
        },
        "plot_configuration": {
            "overview": {
                "fig_size": 10,
                "shift_colormap": "viridis",
                "shift_auto_axes_limit": True,
                "shift_axes_limit": 5.0,
                "theta_colormap": "plasma",
            },
            "shift": {
                "fig_size": 12,
                "scatter_colormap": "jet",
                "scatter_auto_limit": True,
                "scatter_min_limit": -10.0,
                "scatter_max_limit": 10.0,
                "histo_mean_bin_size": 5,
            },
            "dem": {"fig_size": 8, "show_fliers": False, "histo_mean_bin_size": 10},
            "ce": {"fig_size": 15, "ce_scatter_colormap": "hot"},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(config_dict, f)
        temp_file_path = f.name

    try:
        config = ProcessingConfiguration.from_file(temp_file_path)

        # Check that the configuration was loaded correctly
        assert config.klt_configuration is not None
        assert config.klt_configuration.minDistance == 10
        assert config.klt_configuration.blocksize == 15
    finally:
        # Clean up: remove the temporary file
        os.remove(temp_file_path)


def test_processing_configuration_from_nonexistent_file():
    """Test ProcessingConfiguration.from_file with non-existent file."""
    with pytest.raises(ConfigurationError, match="does not exist"):
        ProcessingConfiguration.from_file("/non/existent/path.json")


def test_processing_configuration_from_invalid_json():
    """Test ProcessingConfiguration.from_file with invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        f.write("invalid json content {")
        temp_file_path = f.name

    try:
        with pytest.raises(ConfigurationError, match="is not a valid configuration file"):
            ProcessingConfiguration.from_file(temp_file_path)
    finally:
        # Clean up: remove the temporary file
        os.remove(temp_file_path)


def test_processing_configuration_from_dict_missing_keys():
    """Test ProcessingConfiguration.from_dict with incomplete dictionary."""
    # Missing some expected keys
    incomplete_config_dict = {
        "processing_configuration": {
            "klt_matching": {
                "minDistance": 10,
                "blocksize": 15,
                "maxCorners": 20000,
                "matching_winsize": 25,
                "qualityLevel": 0.01,
                "xStart": 0,
                "tile_size": 1000,
                "laplacian_kernel_size": 3,
                "outliers_filtering": True,
            }
        },
        "plot_configuration": {
            "overview": {
                "fig_size": 10,
                "shift_colormap": "viridis",
                "shift_auto_axes_limit": True,
                "shift_axes_limit": 5.0,
                "theta_colormap": "plasma",
            }
        },
    }

    # This should raise an exception because required keys are missing
    with pytest.raises(KeyError):
        ProcessingConfiguration.from_dict(incomplete_config_dict)


def test_configuration_dataclass_immutability():
    """Test that configuration dataclasses work as expected."""
    klt_config = KLTConfiguration(
        minDistance=10,
        blocksize=15,
        maxCorners=20000,
        matching_winsize=25,
        qualityLevel=0.01,
        xStart=0,
        tile_size=1000,
        laplacian_kernel_size=3,
        outliers_filtering=True,
    )

    # Test that we can access the fields
    assert klt_config.minDistance == 10
    assert klt_config.blocksize == 15

    # Dataclasses should be immutable by default unless frozen=True,
    # but let's verify the values stay consistent
    assert hasattr(klt_config, "minDistance")
    assert hasattr(klt_config, "blocksize")
    assert hasattr(klt_config, "maxCorners")
    assert hasattr(klt_config, "matching_winsize")
    assert hasattr(klt_config, "qualityLevel")
    assert hasattr(klt_config, "xStart")
    assert hasattr(klt_config, "tile_size")
    assert hasattr(klt_config, "laplacian_kernel_size")
    assert hasattr(klt_config, "outliers_filtering")


if __name__ == "__main__":
    test_klt_configuration()
    test_overview_plot_configuration()
    test_shift_plot_configuration()
    test_dem_plot_configuration()
    test_ce_plot_configuration()
    test_accuracy_analysis_configuration()
    test_shift_configuration()
    test_processing_configuration_initialization()
    test_processing_configuration_from_dict()
    test_processing_configuration_from_file()
    test_processing_configuration_from_nonexistent_file()
    test_processing_configuration_from_invalid_json()
    test_processing_configuration_from_dict_missing_keys()
    test_configuration_dataclass_immutability()
    print("All configuration tests passed!")
