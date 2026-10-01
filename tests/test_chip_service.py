#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the key point chips."""

import numpy as np
from osgeo import gdal, osr

from karios.report.chip_service import ChipService, chip_dir_names


def _source(path, geo_transform):
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), 200, 200, 1, gdal.GDT_UInt16)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.SetGeoTransform(geo_transform)
    rng = np.random.default_rng(0)
    dataset.GetRasterBand(1).WriteArray(rng.integers(100, 4000, (200, 200), dtype=np.uint16))
    return dataset


def test_laplacian_chips_are_georeferenced_like_image_chips(tmp_path):
    """gdalbuildvrt skipped the Laplacian chips, written without georeferencing."""
    geo = (600000.0, 10.0, 2.0, 4900000.0, 1.5, -10.0)  # rotated, to check all six terms
    source = _source(tmp_path / "source.tif", geo)
    chip = tmp_path / "REF_100_80.TIFF"

    ChipService()._write_laplacian_chip(source, 72, 52, 3, chip)

    written = gdal.Open(str(chip))
    x_origin, y_origin = gdal.ApplyGeoTransform(geo, 72, 52)
    assert written.GetGeoTransform() == (x_origin, 10.0, 2.0, y_origin, 1.5, -10.0)
    assert osr.SpatialReference(wkt=written.GetProjection()).GetAuthorityCode(None) == "32631"


def test_chip_directories_differ_for_same_file_names():
    """Images with the same file name in two folders get distinct chip directories."""
    assert chip_dir_names("mon.tif", "ref.tif") == ("mon.tif", "ref.tif")
    assert chip_dir_names("B04.jp2", "B04.jp2") == ("B04.jp2_monitored", "B04.jp2_reference")
