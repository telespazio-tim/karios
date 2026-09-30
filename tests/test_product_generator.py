#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the key point products on a rotated WGS 84 grid."""

import json

import pandas as pd
from osgeo import gdal, osr

from karios.api.config import RuntimeConfiguration
from karios.core.image import GdalRasterImage
from karios.report.product_generator import ProductGenerator

# PhiSat scene_0_BC_band_0.tiff: WGS 84 degrees on a rotated, skewed grid
PHISAT_WGS84 = (5.0185, -5.63e-05, -1.22e-05, 43.5735, 2.16e-05, -4.04e-05)


def _reference(path):
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), 64, 64, 1, gdal.GDT_UInt16)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.SetGeoTransform(PHISAT_WGS84)
    dataset = None
    return GdalRasterImage(str(path))


def test_products_keep_the_rotated_georeferencing(tmp_path):
    """Rasters copy the full geotransform; the GeoJSON is in the reference's EPSG, rotated."""
    reference = _reference(tmp_path / "ref.tif")
    points = pd.DataFrame(
        {
            "x0": [10.0, 40.0],
            "y0": [20.0, 5.0],
            "dx": [0.1, -0.2],
            "dy": [0.4, 0.5],
            "score": [0.8, 0.9],
            "radial error": [0.41, 0.54],
            "angle": [76.0, 112.0],
        }
    )
    config = RuntimeConfiguration(
        output_directory=tmp_path,
        gen_kp_mask=True,
        gen_delta_raster=True,
        generate_kp_chips=False,
        enable_large_shift_detection=False,
    )

    products = ProductGenerator(config, points, reference).generate_products()

    rasters = [p for p in products if p.endswith(".tif")]
    assert len(rasters) == 2
    for raster in rasters:
        assert gdal.Open(raster).GetGeoTransform() == PHISAT_WGS84

    geojson = json.loads((tmp_path / "kp_delta.json").read_text(encoding="utf-8"))
    assert geojson["crs"]["properties"]["name"] == "urn:ogc:def:crs:EPSG::4326"
    lon, lat = geojson["features"][0]["geometry"]["coordinates"]
    assert [lon, lat] == list(gdal.ApplyGeoTransform(PHISAT_WGS84, 10.0, 20.0))
