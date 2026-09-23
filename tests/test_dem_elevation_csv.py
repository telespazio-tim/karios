# -*- coding: utf-8 -*-
# Copyright (c) 2026 Telespazio France.
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
"""Tests for the DEM elevation column in the KLT CSV output."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr

import karios
from karios.api.config import RuntimeConfiguration
from karios.api.core import KariosAPI
from karios.core.configuration import ProcessingConfiguration

CONFIG_FILE = Path(karios.__file__).parent / "configuration" / "processing_configuration.json"
SIZE = 32
CSV_NAME = "KLT_matcher_mon_ref.csv"


def _write_raster(path, array, gdal_type):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), SIZE, SIZE, 1, gdal_type)
    dataset.SetGeoTransform((600000.0, 10.0, 0.0, 5000000.0, 0.0, -10.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset = None


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(1, 255, (SIZE, SIZE), dtype=np.uint8)
    _write_raster(tmp_path / "ref.tif", image, gdal.GDT_Byte)
    _write_raster(tmp_path / "mon.tif", image, gdal.GDT_Byte)

    # elevation encodes the pixel position, so each point's expected value is obvious
    dem = (np.arange(SIZE * SIZE).reshape(SIZE, SIZE) * 1.5).astype(np.float32)
    _write_raster(tmp_path / "dem.tif", dem, gdal.GDT_Float32)
    return tmp_path, dem


def _klt_chunks():
    # two chunks, as the matcher streams one DataFrame per tile; scores below the
    # confidence threshold so no ZNCC is computed
    yield pd.DataFrame(
        {"x0": [3.0, 10.6], "y0": [5.0, 20.2], "dx": [0.1, 0.2], "dy": [0.3, 0.4], "score": 0.0}
    )
    yield pd.DataFrame({"x0": [31.0], "y0": [0.0], "dx": [0.5], "dy": [0.6], "score": 0.0})


def _api(out_dir):
    runtime_config = RuntimeConfiguration(
        output_directory=out_dir / "out",
        pixel_size=None,
        title_prefix=None,
        gen_kp_mask=False,
        gen_delta_raster=False,
        generate_kp_chips=False,
        dem_description=None,
        enable_large_shift_detection=False,
    )
    return KariosAPI(ProcessingConfiguration.from_file(CONFIG_FILE), runtime_config)


def _expected_alt(dem, points):
    return dem[points["y0"].astype(int), points["x0"].astype(int)]


def test_csv_has_elevation_column_with_dem(inputs):
    tmp_path, dem = inputs
    api = _api(tmp_path)

    with patch.object(api._klt, "match", return_value=_klt_chunks()):
        result = api.match_images(
            tmp_path / "mon.tif", tmp_path / "ref.tif", dem_file_path=tmp_path / "dem.tif"
        )

    csv = pd.read_csv(tmp_path / "out" / CSV_NAME, sep=";")
    assert len(csv) == 3
    np.testing.assert_array_equal(csv["alt"], _expected_alt(dem, csv))
    np.testing.assert_array_equal(result.points["alt"], csv["alt"])


def test_csv_has_no_elevation_column_without_dem(inputs):
    tmp_path, _ = inputs
    api = _api(tmp_path)

    with patch.object(api._klt, "match", return_value=_klt_chunks()):
        api.match_images(tmp_path / "mon.tif", tmp_path / "ref.tif")

    csv = pd.read_csv(tmp_path / "out" / CSV_NAME, sep=";")
    assert "alt" not in csv.columns


def test_resume_adds_elevation_column_to_existing_csv(inputs):
    tmp_path, dem = inputs
    api = _api(tmp_path)

    # first run without DEM leaves a CSV with no elevation
    with patch.object(api._klt, "match", return_value=_klt_chunks()):
        api.match_images(tmp_path / "mon.tif", tmp_path / "ref.tif")

    with patch.object(api._klt, "match") as match:
        result = api.match_images(
            tmp_path / "mon.tif",
            tmp_path / "ref.tif",
            resume=True,
            dem_file_path=tmp_path / "dem.tif",
        )
        match.assert_not_called()

    csv = pd.read_csv(tmp_path / "out" / CSV_NAME, sep=";")
    np.testing.assert_array_equal(csv["alt"], _expected_alt(dem, csv))
    assert "alt" in result.points.columns
