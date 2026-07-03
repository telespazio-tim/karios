#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for MutualInfoService and related functions."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from karios.core.image import GdalRasterImage
from karios.matcher.mutual_info_service import MutualInfoService, _mutual_info


def test_mutual_info_identical_patches():
    """Test _mutual_info returns 2.0 for identical patches (perfect dependence)."""
    patch = np.arange(57 * 57, dtype=np.float64).reshape(57, 57)
    result = _mutual_info(patch, patch)
    assert abs(result - 2.0) < 1e-10


def test_mutual_info_range():
    """Test _mutual_info result is in [1, 2] for typical correlated patches."""
    rng = np.random.default_rng(42)
    patch1 = rng.random((57, 57))
    patch2 = patch1 + rng.random((57, 57)) * 0.1
    result = _mutual_info(patch1, patch2)
    assert 1.0 <= result <= 2.0


def test_mutual_info_independent_patches():
    """Test _mutual_info is close to 1.0 for statistically independent patches."""
    rng = np.random.default_rng(0)
    patch1 = rng.random((57, 57))
    patch2 = rng.random((57, 57))
    result = _mutual_info(patch1, patch2)
    assert 1.0 <= result < 1.2


def test_mutual_info_uniform_patch_returns_one():
    """Test _mutual_info returns 1.0 when one patch is uniform (zero entropy)."""
    uniform = np.ones((57, 57), dtype=np.float64)
    rng = np.random.default_rng(1)
    patch = rng.random((57, 57))
    result = _mutual_info(uniform, patch)
    # H(uniform)=0, so NMI = (0 + H(Y)) / H(Y) = 1.0
    assert abs(result - 1.0) < 1e-10


def test_mutual_info_both_uniform_returns_nan():
    """Test _mutual_info returns NaN when both patches are uniform (hxy=0)."""
    uniform = np.ones((57, 57), dtype=np.float64)
    result = _mutual_info(uniform, uniform)
    assert np.isnan(result)


def test_mutual_info_custom_bins():
    """Test _mutual_info accepts a custom bins parameter."""
    rng = np.random.default_rng(7)
    patch = rng.random((57, 57))
    result = _mutual_info(patch, patch, bins=64)
    assert abs(result - 2.0) < 1e-10


def test_mutual_info_service_initialization():
    """Test MutualInfoService initialization."""
    service = MutualInfoService()
    assert service._chip_size == 57
    assert service._chip_margin == 28  # (57-1)/2


def test_mutual_info_service_compute_mutual_info():
    """Test MutualInfoService compute_mutual_info method with mocked images."""
    service = MutualInfoService()

    df = pd.DataFrame({"x0": [30, 40], "y0": [30, 40], "dx": [1, -1], "dy": [1, -1]})

    mock_monitored = Mock(spec=GdalRasterImage)
    mock_reference = Mock(spec=GdalRasterImage)
    mock_monitored.clear_cache = Mock()
    mock_reference.clear_cache = Mock()

    with patch.object(df, "apply") as mock_apply:
        mock_apply.return_value = pd.Series([1.6, 1.4], index=df.index)

        result = service.compute_mutual_info(df, mock_monitored, mock_reference)

        mock_apply.assert_called_once()
        assert len(result) == 2
        mock_monitored.clear_cache.assert_called_once()
        mock_reference.clear_cache.assert_called_once()


def test_mutual_info_service_boundary_near_left():
    """Test _compute_mutual_info returns NaN for points too close to top/left boundary."""
    service = MutualInfoService()

    series = pd.Series({"x0": 5.0, "y0": 5.0, "dx": 0.0, "dy": 0.0})

    mock_monitored = Mock(spec=GdalRasterImage)
    mock_reference = Mock(spec=GdalRasterImage)
    mock_monitored.x_size = 200
    mock_monitored.y_size = 200
    mock_reference.x_size = 200
    mock_reference.y_size = 200

    result = service._compute_mutual_info(series, mock_monitored, mock_reference)
    assert np.isnan(result)


def test_mutual_info_service_boundary_near_right():
    """Test _compute_mutual_info returns NaN for points too close to bottom/right boundary."""
    service = MutualInfoService()

    series = pd.Series({"x0": 180.0, "y0": 180.0, "dx": 0.0, "dy": 0.0})

    mock_monitored = Mock(spec=GdalRasterImage)
    mock_reference = Mock(spec=GdalRasterImage)
    mock_monitored.x_size = 200
    mock_monitored.y_size = 200
    mock_reference.x_size = 200
    mock_reference.y_size = 200

    result = service._compute_mutual_info(series, mock_monitored, mock_reference)
    assert np.isnan(result)


def test_mutual_info_service_valid_point():
    """Test _compute_mutual_info returns a value in [1, 2] for a valid interior point."""
    service = MutualInfoService()

    series = pd.Series({"x0": 60.0, "y0": 60.0, "dx": 1.0, "dy": 1.0})

    mock_monitored = Mock(spec=GdalRasterImage)
    mock_reference = Mock(spec=GdalRasterImage)
    mock_monitored.x_size = 200
    mock_monitored.y_size = 200
    mock_reference.x_size = 200
    mock_reference.y_size = 200

    rng = np.random.default_rng(99)
    full_array = rng.random((200, 200))
    mock_monitored.array = full_array
    mock_reference.array = full_array  # identical → NMI should be ~2

    result = service._compute_mutual_info(series, mock_monitored, mock_reference)
    assert 1.0 <= result <= 2.0


def test_mutual_info_service_error_handling():
    """Test _compute_mutual_info returns NaN when _mutual_info raises an exception."""
    service = MutualInfoService()

    series = pd.Series({"x0": 60.0, "y0": 60.0, "dx": 1.0, "dy": 1.0})

    mock_monitored = Mock(spec=GdalRasterImage)
    mock_reference = Mock(spec=GdalRasterImage)
    mock_monitored.x_size = 200
    mock_monitored.y_size = 200
    mock_reference.x_size = 200
    mock_reference.y_size = 200

    test_patch = np.random.rand(57, 57)
    with patch.object(service, "_extract_chip", return_value=test_patch):
        with patch(
            "karios.matcher.mutual_info_service._mutual_info",
            side_effect=Exception("Test error"),
        ):
            result = service._compute_mutual_info(series, mock_monitored, mock_reference)
            assert np.isnan(result)


if __name__ == "__main__":
    test_mutual_info_identical_patches()
    test_mutual_info_range()
    test_mutual_info_independent_patches()
    test_mutual_info_uniform_patch_returns_one()
    test_mutual_info_both_uniform_returns_nan()
    test_mutual_info_custom_bins()
    test_mutual_info_service_initialization()
    test_mutual_info_service_compute_mutual_info()
    test_mutual_info_service_boundary_near_left()
    test_mutual_info_service_boundary_near_right()
    test_mutual_info_service_valid_point()
    test_mutual_info_service_error_handling()
    print("All mutual info service tests passed!")
