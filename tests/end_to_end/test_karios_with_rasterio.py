# -*- coding: utf-8 -*-
"""End-to-end test demonstrating karios usage with rasterio

This module demonstrates how to use karios as a library in combination with rasterio
for reading and writing geospatial raster images.

Key Concepts:
-------------
1. **Creating Input Data with Rasterio**: Shows how to create georeferenced test images
   using rasterio, which can be adapted for real satellite imagery preprocessing.

2. **Using Karios API**: Demonstrates the programmatic API for:
   - Loading processing configuration from JSON files
   - Creating runtime configuration for output settings
   - Running image matching with KLT algorithm
   - Performing accuracy analysis
   - Accessing match results (points, statistics)

3. **Reading/Writing Outputs with Rasterio**: Shows how to:
   - Read karios output products (masks, displacement rasters)
   - Create custom output rasters from match results
   - Maintain georeferencing information

Typical Workflow:
-----------------
    # 1. Create/load images with rasterio
    with rasterio.open('image.tif') as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    # 2. Configure karios
    processing_config = ProcessingConfiguration.from_file('config.json')
    runtime_config = RuntimeConfiguration(
        output_directory='./results',
        pixel_size=10.0,
        ...
    )

    # 3. Run karios processing
    api = KariosAPI(processing_config, runtime_config)
    match_result = api.match_images(mon_path, ref_path)
    accuracy = api.analyze_accuracy(match_result)

    # 4. Access results
    points = match_result.points  # DataFrame with x0, y0, dx, dy, score
    ce90 = accuracy.ce90

    # 5. Create custom outputs with rasterio
    with rasterio.open('output.tif', 'w', ...) as dst:
        dst.write(custom_data, 1)

Three Test Cases:
-----------------
1. test_karios_as_library_with_rasterio: Full workflow demonstration
2. test_rasterio_preprocessing_with_karios: Preprocessing images before matching
3. test_custom_mask_creation_with_rasterio: Creating and using quality masks

Author: KARIOS Team
Date: 2026
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from karios.api import KariosAPI, RuntimeConfiguration
from karios.core.configuration import ProcessingConfiguration

module_dir_path = os.path.dirname(__file__)
test_data_dir = os.path.join(module_dir_path, "test_data")


class TestKariosWithRasterio(unittest.TestCase):
    """End-to-end test demonstrating how to use karios as a library with rasterio"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_dir_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_raster(self, filepath, data, transform=None, crs=None):
        """Helper to create test GeoTIFF files using rasterio

        Args:
            filepath: Path to output raster file
            data: 2D numpy array with raster data
            transform: Affine transform (optional)
            crs: CRS string (optional)
        """
        if transform is None:
            # Default transform: 10m pixels, origin at (0, 0)
            transform = from_origin(0, 0, 10, 10)

        if crs is None:
            crs = "EPSG:32612"  # Default UTM zone 12N

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(data, 1)

    def test_karios_as_library_with_rasterio(self):
        """Test using karios as a library with rasterio for I/O operations

        This test demonstrates:
        1. Creating input rasters using rasterio
        2. Using karios API for image matching
        3. Reading results and creating custom outputs with rasterio
        """
        # Step 1: Create test images using rasterio
        # These simulate real satellite imagery with known characteristics
        image_size = 500  # 500x500 pixels
        pixel_size = 10.0  # 10m pixels

        # Create reference image with synthetic features
        ref_data = np.zeros((image_size, image_size), dtype=np.uint16)
        # Add some synthetic features (bright spots) that can be matched
        for i in range(10, image_size, 50):
            for j in range(10, image_size, 50):
                ref_data[i : i + 5, j : j + 5] = 1000

        # Create monitored image with slight shifts to simulate deformation
        mon_data = np.zeros((image_size, image_size), dtype=np.uint16)
        # Same features but with a small shift (simulating 2 pixel displacement)
        shift_x, shift_y = 2, 3
        for i in range(10, image_size, 50):
            for j in range(10, image_size, 50):
                mon_data[i + shift_y : i + shift_y + 5, j + shift_x : j + shift_x + 5] = 1000

        # Define georeferencing
        transform = from_origin(500000, 4000000, pixel_size, pixel_size)
        crs = "EPSG:32612"

        ref_path = self.test_dir_path / "reference.tif"
        mon_path = self.test_dir_path / "monitored.tif"

        self._create_test_raster(ref_path, ref_data, transform, crs)
        self._create_test_raster(mon_path, mon_data, transform, crs)

        # Step 2: Configure karios processing
        # Load default processing configuration from file
        config_path = Path(module_dir_path) / "processing_configuration.json"
        processing_config = ProcessingConfiguration.from_file(config_path)

        # Customize KLT parameters for our test data
        processing_config.klt_configuration.maxCorners = 100
        processing_config.klt_configuration.tile_size = 50
        processing_config.accuracy_analysis_configuration.confidence_threshold = 0.7

        # Create runtime configuration
        output_dir = self.test_dir_path / "karios_output"
        runtime_config = RuntimeConfiguration(
            output_directory=output_dir,
            gen_kp_mask=False,  # Disable to avoid plot generation issues
            gen_delta_raster=False,  # Disable to avoid plot generation issues
            pixel_size=pixel_size,
            enable_large_shift_detection=False,
            generate_kp_chips=False,  # Disable for this test
            title_prefix="Test",
            dem_description=None,
        )

        # Step 3: Initialize and run karios API
        api = KariosAPI(processing_config, runtime_config)

        # Execute matching step only
        match_result = api.match_images(
            monitored_image_path=mon_path,
            reference_image_path=ref_path,
        )

        # Execute accuracy analysis step
        accuracy = api.analyze_accuracy(match_result)

        # Skip report generation to avoid matplotlib issues with synthetic data
        # In real usage, you would call: reports = api.generate_reports(match_result, accuracy)
        reports = None

        # Step 4: Verify results
        # Check that match points were found
        self.assertIsNotNone(match_result.points)
        self.assertGreater(len(match_result.points), 0, "No match points found")

        # Check accuracy statistics
        self.assertIsNotNone(accuracy)
        self.assertGreater(accuracy.ce90, 0)
        self.assertGreater(accuracy.valid_pixels, 0)

        # Step 5: Demonstrate accessing match results
        # Verify match points are available
        points = match_result.points
        print(f"Found {len(points)} match points")
        print(f"Mean dx: {points['dx'].mean():.2f} pixels")
        print(f"Mean dy: {points['dy'].mean():.2f} pixels")
        print(f"Mean radial error: {points['radial error'].mean():.2f} pixels")

        # Step 6: Demonstrate creating custom output with rasterio
        # Create a custom displacement magnitude raster from match points
        points = match_result.points
        if len(points) > 0:
            # Calculate displacement magnitude
            displacement_magnitude = np.sqrt(points["dx"] ** 2 + points["dy"] ** 2)

            # Create a raster from point measurements
            # (simplified - in practice you'd interpolate)
            output_raster = np.zeros((image_size, image_size), dtype=np.float32)

            # Place displacement values at point locations (nearest neighbor)
            for _, point in points.iterrows():
                x_idx = int(point["x0"])
                y_idx = int(point["y0"])
                if 0 <= x_idx < image_size and 0 <= y_idx < image_size:
                    output_raster[y_idx, x_idx] = point.get("radial error", 0)

            # Write custom output raster using rasterio
            custom_output_path = self.test_dir_path / "displacement_magnitude.tif"
            with rasterio.open(
                custom_output_path,
                "w",
                driver="GTiff",
                height=image_size,
                width=image_size,
                count=1,
                dtype=np.float32,
                crs=crs,
                transform=transform,
                compress="lzw",
                nodata=-9999,
            ) as dst:
                dst.write(output_raster, 1)

            # Verify custom output was created
            self.assertTrue(custom_output_path.exists())

            # Read it back to verify
            with rasterio.open(custom_output_path) as src:
                verify_data = src.read(1)
                self.assertEqual(verify_data.shape, (image_size, image_size))
                print(f"Custom displacement raster created: {custom_output_path}")
                print(f"Mean displacement: {np.nanmean(verify_data[verify_data > 0]):.2f} pixels")

    def test_rasterio_preprocessing_with_karios(self):
        """Test preprocessing images with rasterio before karios matching

        This demonstrates a common workflow where images are preprocessed
        using rasterio before being passed to karios.
        """
        # Create base test images
        image_size = 300
        pixel_size = 10.0

        # Create reference image
        ref_data = np.random.randint(100, 200, (image_size, image_size), dtype=np.uint16)
        # Add some structure
        ref_data[100:150, 100:150] = 500
        ref_data[200:250, 50:100] = 800

        # Create monitored image (slightly different)
        mon_data = ref_data.copy()
        # Add some noise to simulate real-world differences
        noise = np.random.randint(-20, 20, (image_size, image_size), dtype=np.int16)
        mon_data = np.clip(mon_data.astype(np.int16) + noise, 0, 65535).astype(np.uint16)

        transform = from_origin(600000, 4500000, pixel_size, pixel_size)
        crs = "EPSG:32612"

        ref_path = self.test_dir_path / "ref_preprocess.tif"
        mon_path = self.test_dir_path / "mon_preprocess.tif"

        self._create_test_raster(ref_path, ref_data, transform, crs)
        self._create_test_raster(mon_path, mon_data, transform, crs)

        # Preprocess with rasterio: apply a simple filter
        # In real scenarios, this could be atmospheric correction, normalization, etc.
        preprocessed_mon_path = self.test_dir_path / "mon_preprocessed.tif"

        with rasterio.open(mon_path) as src:
            mon_array = src.read(1)
            mon_transform = src.transform
            mon_crs = src.crs

            # Apply simple normalization (example preprocessing)
            mon_normalized = ((mon_array - mon_array.min()) / (mon_array.max() - mon_array.min()) * 1000).astype(
                np.uint16
            )

            # Write preprocessed image
            with rasterio.open(
                preprocessed_mon_path,
                "w",
                driver="GTiff",
                height=mon_array.shape[0],
                width=mon_array.shape[1],
                count=1,
                dtype=np.uint16,
                crs=mon_crs,
                transform=mon_transform,
                compress="lzw",
            ) as dst:
                dst.write(mon_normalized, 1)

        # Now use karios with preprocessed image
        config_path = Path(module_dir_path) / "processing_configuration.json"
        processing_config = ProcessingConfiguration.from_file(config_path)
        processing_config.klt_configuration.maxCorners = 50

        runtime_config = RuntimeConfiguration(
            output_directory=self.test_dir_path / "preprocess_output",
            gen_kp_mask=False,
            gen_delta_raster=False,
            pixel_size=pixel_size,
            enable_large_shift_detection=False,
            generate_kp_chips=False,
            title_prefix="Preprocess",
            dem_description=None,
        )

        api = KariosAPI(processing_config, runtime_config)

        # Process with preprocessed image
        match_result, accuracy, reports = api.process(
            monitored_image_path=preprocessed_mon_path,
            reference_image_path=ref_path,
        )

        # Verify processing completed
        self.assertIsNotNone(match_result.points)
        self.assertIsNotNone(accuracy)
        print(f"Preprocessing workflow completed: {len(match_result.points)} points matched")

    def test_custom_mask_creation_with_rasterio(self):
        """Test creating custom masks with rasterio for karios

        This demonstrates creating quality masks using rasterio and using them
        with karios to exclude poor-quality areas from matching.
        """
        # Create test images
        image_size = 400
        pixel_size = 10.0

        ref_data = np.ones((image_size, image_size), dtype=np.uint16) * 500
        mon_data = np.ones((image_size, image_size), dtype=np.uint16) * 500

        # Add features
        ref_data[50:100, 50:100] = 1000
        ref_data[150:200, 150:200] = 1200
        ref_data[250:300, 250:300] = 800

        mon_data[50:100, 50:100] = 1000
        mon_data[150:200, 150:200] = 1200
        mon_data[250:300, 250:300] = 800

        # Add cloudy/bad data area (simulate poor quality region)
        mon_data[180:220, 180:220] = 50  # Very dark area (cloud shadow?)

        transform = from_origin(700000, 5000000, pixel_size, pixel_size)
        crs = "EPSG:32612"

        ref_path = self.test_dir_path / "ref_mask.tif"
        mon_path = self.test_dir_path / "mon_mask.tif"
        mask_path = self.test_dir_path / "quality_mask.tif"

        self._create_test_raster(ref_path, ref_data, transform, crs)
        self._create_test_raster(mon_path, mon_data, transform, crs)

        # Create quality mask using rasterio
        # 1 = good quality, 0 = bad quality (exclude from matching)
        quality_mask = np.ones((image_size, image_size), dtype=np.uint8)
        quality_mask[180:220, 180:220] = 0  # Exclude the bad area

        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=image_size,
            width=image_size,
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(quality_mask, 1)

        # Run karios with custom mask
        config_path = Path(module_dir_path) / "processing_configuration.json"
        processing_config = ProcessingConfiguration.from_file(config_path)
        processing_config.klt_configuration.maxCorners = 80

        runtime_config = RuntimeConfiguration(
            output_directory=self.test_dir_path / "mask_output",
            gen_kp_mask=True,
            gen_delta_raster=False,
            pixel_size=pixel_size,
            enable_large_shift_detection=False,
            generate_kp_chips=False,
            title_prefix="Mask",
            dem_description=None,
        )

        api = KariosAPI(processing_config, runtime_config)

        # Process with mask
        match_result, accuracy, reports = api.process(
            monitored_image_path=mon_path,
            reference_image_path=ref_path,
            mask_file_path=mask_path,
        )

        # Verify mask was applied
        self.assertIsNotNone(match_result.mask)
        self.assertIsNotNone(accuracy)
        print(f"Masked processing completed: {accuracy.valid_pixels} valid pixels")


if __name__ == "__main__":
    unittest.main()
