#!/usr/bin/env python3
"""Unit tests for GdalRasterImage class."""

import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
from karios.core.image import GdalRasterImage, GdalError, shift_image, get_image_resolution, open_gdal_dataset


def test_gdal_raster_image_initialization():
    """Test GdalRasterImage initialization."""
    with patch('karios.core.image.open_gdal_dataset') as mock_open_gdal:
        # Mock dataset properties
        mock_dataset = Mock()
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)  # (x_min, x_res, 0, y_max, 0, y_res)
        mock_dataset.RasterXSize = 100
        mock_dataset.RasterYSize = 100
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_dataset.GetSpatialRef.return_value = Mock()
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_dataset
        mock_open_gdal.return_value = mock_context

        image = GdalRasterImage("/fake/path/image.tif")
        
        assert image.filepath == "/fake/path/image.tif"
        assert image.x_size == 100
        assert image.y_size == 100
        assert image.x_res == 1
        assert image.y_res == -1
        assert image.x_min == 0
        assert image.y_max == 0


def test_gdal_raster_image_properties():
    """Test GdalRasterImage property access."""
    with patch('karios.core.image.open_gdal_dataset') as mock_open_gdal:
        mock_dataset = Mock()
        mock_dataset.GetGeoTransform.return_value = (10, 2, 0, 20, 0, -2)
        mock_dataset.RasterXSize = 50
        mock_dataset.RasterYSize = 60
        mock_dataset.GetProjection.return_value = "EPSG:32633"
        mock_dataset.GetSpatialRef.return_value = Mock()
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_dataset
        mock_open_gdal.return_value = mock_context

        image = GdalRasterImage("/fake/path/image.tif")
        
        assert image.geo_transform == (10, 2, 0, 20, 0, -2)
        assert image.x_res == 2
        assert image.y_res == -2
        assert image.x_min == 10
        assert image.y_max == 20
        assert image.x_size == 50
        assert image.y_size == 60
        assert image.have_pixel_resolution() is True


def test_gdal_raster_image_read():
    """Test reading sub-arrays from image."""
    with patch('karios.core.image.open_gdal_dataset') as mock_open_gdal:
        mock_dataset = Mock()
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_dataset.RasterXSize = 100
        mock_dataset.RasterYSize = 100
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_dataset.GetSpatialRef.return_value = Mock()
        
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = np.ones((20, 20))
        mock_dataset.GetRasterBand.return_value = mock_band
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_dataset
        mock_open_gdal.return_value = mock_context

        image = GdalRasterImage("/fake/path/image.tif")
        result = image.read(1, 10, 10, 20, 20)
        
        assert result.shape == (20, 20)
        mock_dataset.GetRasterBand.assert_called_with(1)
        mock_band.ReadAsArray.assert_called_with(10, 10, 20, 20)


def test_gdal_raster_image_array():
    """Test array property with caching."""
    with patch('karios.core.image.open_gdal_dataset') as mock_open_gdal:
        mock_dataset = Mock()
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_dataset.RasterXSize = 100
        mock_dataset.RasterYSize = 100
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_dataset.GetSpatialRef.return_value = Mock()
        
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = np.random.rand(100, 100)
        mock_dataset.GetRasterBand.return_value = mock_band
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_dataset
        mock_open_gdal.return_value = mock_context

        image = GdalRasterImage("/fake/path/image.tif")
        array1 = image.array
        array2 = image.array
        
        # Check that array was cached (band read only once)
        assert mock_band.ReadAsArray.call_count == 1
        assert array1 is array2  # Should be same object due to caching


def test_gdal_raster_image_clear_cache():
    """Test clearing cached array."""
    with patch('karios.core.image.open_gdal_dataset') as mock_open_gdal:
        mock_dataset = Mock()
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_dataset.RasterXSize = 100
        mock_dataset.RasterYSize = 100
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_dataset.GetSpatialRef.return_value = Mock()
        
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = np.random.rand(100, 100)
        mock_dataset.GetRasterBand.return_value = mock_band
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_dataset
        mock_open_gdal.return_value = mock_context

        image = GdalRasterImage("/fake/path/image.tif")
        _ = image.array  # Load array to cache
        assert image._array is not None
        
        image.clear_cache()
        assert image._array is None


def test_image_shift_function():
    """Test the shift_image function with various offsets."""
    # Original image
    original = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.float32)
    
    # Test no offset
    result = shift_image(original, 0, 0)
    np.testing.assert_array_equal(result, original)
    
    # Test positive x offset (shift right)
    result = shift_image(original, 0, 1)
    expected = np.array([[2, 3, 0],
                         [5, 6, 0],
                         [8, 9, 0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    
    # Test negative x offset (shift left)
    result = shift_image(original, 0, -1)
    expected = np.array([[0, 1, 2],
                         [0, 4, 5],
                         [0, 7, 8]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    
    # Test positive y offset (shift up - image content moves up, so bottom gets zeros)
    result = shift_image(original, 1, 0)
    expected = np.array([[4, 5, 6],
                         [7, 8, 9],
                         [0, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    
    # Test negative y offset (shift down - image content moves down, so top gets zeros)
    result = shift_image(original, -1, 0)
    expected = np.array([[0, 0, 0],
                         [1, 2, 3],
                         [4, 5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

    # Test combined offsets
    result = shift_image(original, 1, 1)  # shift up by 1 and right by 1
    expected = np.array([[5, 6, 0],
                         [8, 9, 0],
                         [0, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_get_image_resolution():
    """Test get_image_resolution function with different scenarios."""
    from unittest.mock import patch, MagicMock
    import logging

    # Create mock images with different resolution scenarios
    mock_monitored = Mock()
    mock_reference = Mock()

    # Set up required attributes
    mock_monitored.have_pixel_resolution.return_value = True
    mock_monitored.x_res = 10.0
    mock_monitored.y_res = -10.0  # Add y_res as it's needed for the abs comparison
    mock_reference.x_res = 10.0

    result = get_image_resolution(mock_monitored, mock_reference)
    assert result == 10.0

    # Case 2: Monitored image has resolution, default provided (should ignore default)
    result = get_image_resolution(mock_monitored, mock_reference, default_value=5.0)
    assert result == 10.0

    # Case 3: Monitored image has no resolution, default provided
    mock_monitored2 = Mock()
    mock_monitored2.have_pixel_resolution.return_value = False
    result = get_image_resolution(mock_monitored2, mock_reference, default_value=15.0)
    assert result == 15.0

    # Case 4: Monitored image has no resolution, no default provided
    result = get_image_resolution(mock_monitored2, mock_reference, default_value=None)
    assert result is None


def test_gdal_error_exception():
    """Test that GdalError is properly raised."""
    with pytest.raises(GdalError):
        raise GdalError("Test GDAL error")


def test_open_gdal_dataset_context_manager():
    """Test the open_gdal_dataset context manager."""
    with patch('karios.core.image.gdal') as mock_gdal:
        mock_dataset = Mock()
        mock_gdal.Open.return_value = mock_dataset
        
        with open_gdal_dataset("/fake/path.tif") as ds:
            assert ds is mock_dataset
            
        # Verify dataset was opened
        mock_gdal.Open.assert_called_once_with("/fake/path.tif")


def test_open_gdal_dataset_context_manager_failure():
    """Test the open_gdal_dataset context manager when opening fails."""
    with patch('karios.core.image.gdal') as mock_gdal:
        mock_gdal.Open.return_value = None  # Simulate failure
        
        with pytest.raises(GdalError):
            with open_gdal_dataset("/fake/path.tif"):
                pass


def test_is_compatible_with():
    """Test image compatibility check."""
    with patch('karios.core.image.open_gdal_dataset') as mock_open_gdal:
        mock_dataset = Mock()
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_dataset.RasterXSize = 100
        mock_dataset.RasterYSize = 100
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_spatial_ref = Mock()
        mock_spatial_ref.IsSame.return_value = True
        mock_dataset.GetSpatialRef.return_value = mock_spatial_ref
        
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_dataset
        mock_open_gdal.return_value = mock_context

        image1 = GdalRasterImage("/fake/path/image1.tif")
        image2 = GdalRasterImage("/fake/path/image2.tif")
        
        # Mock the other image's spatial reference for comparison
        image2.spatial_ref = mock_spatial_ref
        
        assert image1.is_compatible_with(image2) is True


if __name__ == "__main__":
    test_gdal_raster_image_initialization()
    test_gdal_raster_image_properties()
    test_gdal_raster_image_read()
    test_gdal_raster_image_array()
    test_gdal_raster_image_clear_cache()
    test_image_shift_function()
    test_get_image_resolution()
    test_gdal_error_exception()
    test_open_gdal_dataset_context_manager()
    test_open_gdal_dataset_context_manager_failure()
    test_is_compatible_with()
    print("All GdalRasterImage tests passed!")