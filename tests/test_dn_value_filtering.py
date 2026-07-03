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
"""Tests for DN value filtering functionality."""

import numpy as np
import pandas as pd
import pytest

from karios.api.config import RuntimeConfiguration
from karios.api.core import KariosAPI
from karios.core.configuration import ProcessingConfiguration
from karios.core.image import GdalRasterImage


class TestDNValueFiltering:
    """Test suite for DN value filtering functionality."""

    def test_filter_no_values_none(self, tmp_path):
        """Test that no filtering occurs when no_values is None."""
        # Create test data
        points = pd.DataFrame(
            {"x0": [1, 2, 3, 4, 5], "y0": [1, 2, 3, 4, 5], "dx": [0.1, 0.2, 0.3, 0.4, 0.5], "dy": [0.1, 0.2, 0.3, 0.4, 0.5], "score": [0.9, 0.8, 0.7, 0.6, 0.5]}
        )

        # Create simple test images
        ref_array = np.ones((10, 10), dtype=np.uint8)
        mon_array = np.ones((10, 10), dtype=np.uint8)

        ref_img_path = tmp_path / "ref.tif"
        mon_img_path = tmp_path / "mon.tif"

        from osgeo import gdal

        driver = gdal.GetDriverByName("GTiff")
        ref_ds = driver.Create(str(ref_img_path), 10, 10, 1, gdal.GDT_Byte)
        ref_ds.GetRasterBand(1).WriteArray(ref_array)
        ref_ds = None

        mon_ds = driver.Create(str(mon_img_path), 10, 10, 1, gdal.GDT_Byte)
        mon_ds.GetRasterBand(1).WriteArray(mon_array)
        mon_ds = None

        ref_img = GdalRasterImage(str(ref_img_path))
        mon_img = GdalRasterImage(str(mon_img_path))

        # Create API instance
        proc_config = ProcessingConfiguration()
        runtime_config = RuntimeConfiguration(
            output_directory=tmp_path,
            pixel_size=None,
            title_prefix=None,
            gen_kp_mask=False,
            gen_delta_raster=False,
            generate_kp_chips=False,
            dem_description=None,
            enable_large_shift_detection=False,
            no_values=None,
        )

        api = KariosAPI(proc_config, runtime_config)

        # Test filtering
        filtered_points = api._filter_by_dn_values(points, mon_img, ref_img, None)

        # Should return original points unchanged
        assert len(filtered_points) == len(points)
        pd.testing.assert_frame_equal(filtered_points, points)

    def test_filter_no_values_empty(self, tmp_path):
        """Test that no filtering occurs when no_values is empty list."""
        # Create test data
        points = pd.DataFrame(
            {"x0": [1, 2, 3, 4, 5], "y0": [1, 2, 3, 4, 5], "dx": [0.1, 0.2, 0.3, 0.4, 0.5], "dy": [0.1, 0.2, 0.3, 0.4, 0.5], "score": [0.9, 0.8, 0.7, 0.6, 0.5]}
        )

        # Create simple test images
        ref_array = np.ones((10, 10), dtype=np.uint8)
        mon_array = np.ones((10, 10), dtype=np.uint8)

        ref_img_path = tmp_path / "ref.tif"
        mon_img_path = tmp_path / "mon.tif"

        from osgeo import gdal

        driver = gdal.GetDriverByName("GTiff")
        ref_ds = driver.Create(str(ref_img_path), 10, 10, 1, gdal.GDT_Byte)
        ref_ds.GetRasterBand(1).WriteArray(ref_array)
        ref_ds = None

        mon_ds = driver.Create(str(mon_img_path), 10, 10, 1, gdal.GDT_Byte)
        mon_ds.GetRasterBand(1).WriteArray(mon_array)
        mon_ds = None

        ref_img = GdalRasterImage(str(ref_img_path))
        mon_img = GdalRasterImage(str(mon_img_path))

        # Create API instance
        proc_config = ProcessingConfiguration()
        runtime_config = RuntimeConfiguration(
            output_directory=tmp_path,
            pixel_size=None,
            title_prefix=None,
            gen_kp_mask=False,
            gen_delta_raster=False,
            generate_kp_chips=False,
            dem_description=None,
            enable_large_shift_detection=False,
            no_values=[],
        )

        api = KariosAPI(proc_config, runtime_config)

        # Test filtering
        filtered_points = api._filter_by_dn_values(points, mon_img, ref_img, [])

        # Should return original points unchanged
        assert len(filtered_points) == len(points)
        pd.testing.assert_frame_equal(filtered_points, points)

    def test_filter_zero_values(self, tmp_path):
        """Test filtering out key points with zero DN values."""
        # Create test data - points at positions with different values
        points = pd.DataFrame(
            {
                "x0": [1, 2, 3, 4],
                "y0": [1, 2, 3, 4],
                "dx": [0.1, 0.2, 0.3, 0.4],
                "dy": [0.1, 0.2, 0.3, 0.4],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )

        # Create test images with some zero values
        ref_array = np.zeros((10, 10), dtype=np.uint8)
        ref_array[1, 1] = 100  # Non-zero
        ref_array[2, 2] = 0    # Zero - should be filtered
        ref_array[3, 3] = 150  # Non-zero
        ref_array[4, 4] = 0    # Zero - should be filtered

        mon_array = np.ones((10, 10), dtype=np.uint8) * 50  # All non-zero

        ref_img_path = tmp_path / "ref.tif"
        mon_img_path = tmp_path / "mon.tif"

        from osgeo import gdal

        driver = gdal.GetDriverByName("GTiff")
        ref_ds = driver.Create(str(ref_img_path), 10, 10, 1, gdal.GDT_Byte)
        ref_ds.GetRasterBand(1).WriteArray(ref_array)
        ref_ds = None

        mon_ds = driver.Create(str(mon_img_path), 10, 10, 1, gdal.GDT_Byte)
        mon_ds.GetRasterBand(1).WriteArray(mon_array)
        mon_ds = None

        ref_img = GdalRasterImage(str(ref_img_path))
        mon_img = GdalRasterImage(str(mon_img_path))

        # Create API instance
        proc_config = ProcessingConfiguration()
        runtime_config = RuntimeConfiguration(
            output_directory=tmp_path,
            pixel_size=None,
            title_prefix=None,
            gen_kp_mask=False,
            gen_delta_raster=False,
            generate_kp_chips=False,
            dem_description=None,
            enable_large_shift_detection=False,
            no_values=[0],
        )

        api = KariosAPI(proc_config, runtime_config)

        # Test filtering
        filtered_points = api._filter_by_dn_values(points, mon_img, ref_img, [0])

        # Should keep only points at (1,1) and (3,3)
        assert len(filtered_points) == 2
        assert list(filtered_points["x0"]) == [1, 3]
        assert list(filtered_points["y0"]) == [1, 3]

    def test_filter_multiple_values(self, tmp_path):
        """Test filtering out multiple DN values."""
        # Create test data
        points = pd.DataFrame(
            {
                "x0": [1, 2, 3, 4, 5],
                "y0": [1, 2, 3, 4, 5],
                "dx": [0.1, 0.2, 0.3, 0.4, 0.5],
                "dy": [0.1, 0.2, 0.3, 0.4, 0.5],
                "score": [0.9, 0.8, 0.7, 0.6, 0.5],
            }
        )

        # Create test images with different excluded values
        ref_array = np.ones((10, 10), dtype=np.uint8) * 50
        ref_array[1, 1] = 0    # Should be filtered (no_value=0)
        ref_array[2, 2] = 100  # Should be filtered (no_value=100)
        ref_array[3, 3] = 150  # Keep
        ref_array[4, 4] = 255  # Should be filtered (no_value=255)
        ref_array[5, 5] = 75   # Keep

        mon_array = np.ones((10, 10), dtype=np.uint8) * 50  # All same value

        ref_img_path = tmp_path / "ref.tif"
        mon_img_path = tmp_path / "mon.tif"

        from osgeo import gdal

        driver = gdal.GetDriverByName("GTiff")
        ref_ds = driver.Create(str(ref_img_path), 10, 10, 1, gdal.GDT_Byte)
        ref_ds.GetRasterBand(1).WriteArray(ref_array)
        ref_ds = None

        mon_ds = driver.Create(str(mon_img_path), 10, 10, 1, gdal.GDT_Byte)
        mon_ds.GetRasterBand(1).WriteArray(mon_array)
        mon_ds = None

        ref_img = GdalRasterImage(str(ref_img_path))
        mon_img = GdalRasterImage(str(mon_img_path))

        # Create API instance
        proc_config = ProcessingConfiguration()
        runtime_config = RuntimeConfiguration(
            output_directory=tmp_path,
            pixel_size=None,
            title_prefix=None,
            gen_kp_mask=False,
            gen_delta_raster=False,
            generate_kp_chips=False,
            dem_description=None,
            enable_large_shift_detection=False,
            no_values=[0, 100, 255],
        )

        api = KariosAPI(proc_config, runtime_config)

        # Test filtering
        filtered_points = api._filter_by_dn_values(points, mon_img, ref_img, [0, 100, 255])

        # Should keep only points at (3,3) and (5,5)
        assert len(filtered_points) == 2
        assert list(filtered_points["x0"]) == [3, 5]
        assert list(filtered_points["y0"]) == [3, 5]

    def test_filter_monitored_image_values(self, tmp_path):
        """Test filtering based on monitored image DN values."""
        # Create test data
        points = pd.DataFrame(
            {
                "x0": [1, 2, 3, 4],
                "y0": [1, 2, 3, 4],
                "dx": [0.1, 0.2, 0.3, 0.4],
                "dy": [0.1, 0.2, 0.3, 0.4],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )

        # Reference image - all good values
        ref_array = np.ones((10, 10), dtype=np.uint8) * 100

        # Monitored image - some zeros
        mon_array = np.ones((10, 10), dtype=np.uint8) * 50
        mon_array[1, 1] = 0   # Should be filtered
        mon_array[2, 2] = 50  # Keep
        mon_array[3, 3] = 0   # Should be filtered
        mon_array[4, 4] = 50  # Keep

        ref_img_path = tmp_path / "ref.tif"
        mon_img_path = tmp_path / "mon.tif"

        from osgeo import gdal

        driver = gdal.GetDriverByName("GTiff")
        ref_ds = driver.Create(str(ref_img_path), 10, 10, 1, gdal.GDT_Byte)
        ref_ds.GetRasterBand(1).WriteArray(ref_array)
        ref_ds = None

        mon_ds = driver.Create(str(mon_img_path), 10, 10, 1, gdal.GDT_Byte)
        mon_ds.GetRasterBand(1).WriteArray(mon_array)
        mon_ds = None

        ref_img = GdalRasterImage(str(ref_img_path))
        mon_img = GdalRasterImage(str(mon_img_path))

        # Create API instance
        proc_config = ProcessingConfiguration()
        runtime_config = RuntimeConfiguration(
            output_directory=tmp_path,
            pixel_size=None,
            title_prefix=None,
            gen_kp_mask=False,
            gen_delta_raster=False,
            generate_kp_chips=False,
            dem_description=None,
            enable_large_shift_detection=False,
            no_values=[0],
        )

        api = KariosAPI(proc_config, runtime_config)

        # Test filtering
        filtered_points = api._filter_by_dn_values(points, mon_img, ref_img, [0])

        # Should keep only points at (2,2) and (4,4)
        assert len(filtered_points) == 2
        assert list(filtered_points["x0"]) == [2, 4]
        assert list(filtered_points["y0"]) == [2, 4]

    def test_filter_both_images(self, tmp_path):
        """Test filtering when both images can have excluded values."""
        # Create test data
        points = pd.DataFrame(
            {
                "x0": [1, 2, 3, 4],
                "y0": [1, 2, 3, 4],
                "dx": [0.1, 0.2, 0.3, 0.4],
                "dy": [0.1, 0.2, 0.3, 0.4],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )

        # Reference image - some zeros
        ref_array = np.ones((10, 10), dtype=np.uint8) * 100
        ref_array[1, 1] = 0   # Should be filtered (ref=0)
        ref_array[2, 2] = 100 # Keep
        ref_array[3, 3] = 100 # Keep
        ref_array[4, 4] = 100 # Keep

        # Monitored image - some zeros at different positions
        mon_array = np.ones((10, 10), dtype=np.uint8) * 50
        mon_array[1, 1] = 50  # Keep (ref is 0, but we test combined)
        mon_array[2, 2] = 0   # Should be filtered (mon=0)
        mon_array[3, 3] = 50  # Keep
        mon_array[4, 4] = 0   # Should be filtered (mon=0)

        ref_img_path = tmp_path / "ref.tif"
        mon_img_path = tmp_path / "mon.tif"

        from osgeo import gdal

        driver = gdal.GetDriverByName("GTiff")
        ref_ds = driver.Create(str(ref_img_path), 10, 10, 1, gdal.GDT_Byte)
        ref_ds.GetRasterBand(1).WriteArray(ref_array)
        ref_ds = None

        mon_ds = driver.Create(str(mon_img_path), 10, 10, 1, gdal.GDT_Byte)
        mon_ds.GetRasterBand(1).WriteArray(mon_array)
        mon_ds = None

        ref_img = GdalRasterImage(str(ref_img_path))
        mon_img = GdalRasterImage(str(mon_img_path))

        # Create API instance
        proc_config = ProcessingConfiguration()
        runtime_config = RuntimeConfiguration(
            output_directory=tmp_path,
            pixel_size=None,
            title_prefix=None,
            gen_kp_mask=False,
            gen_delta_raster=False,
            generate_kp_chips=False,
            dem_description=None,
            enable_large_shift_detection=False,
            no_values=[0],
        )

        api = KariosAPI(proc_config, runtime_config)

        # Test filtering
        filtered_points = api._filter_by_dn_values(points, mon_img, ref_img, [0])

        # Should keep only point at (3,3) - only position where neither image has 0
        assert len(filtered_points) == 1
        assert list(filtered_points["x0"]) == [3]
        assert list(filtered_points["y0"]) == [3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
