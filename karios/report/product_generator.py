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
"""Module having class to create "products".
Products are (geo) raster or vector files created from KP and their properties (dx, dy, etc ...)
"""

import json
import logging
import os

import numpy as np
from osgeo import gdal
from pandas import DataFrame, Series

from karios.api.config import RuntimeConfiguration
from karios.core.image import GdalRasterImage

logger = logging.getLogger(__name__)


def _row_slices(sorted_y: np.ndarray):
    """Yield (row, start, end) for each unique row in a sorted integer y-index array."""
    if len(sorted_y) == 0:
        return
    unique, starts = np.unique(sorted_y, return_index=True)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(sorted_y)
    for row, start, end in zip(unique, starts, ends):
        yield int(row), int(start), int(end)


def _to_feature(series: Series, geo_transform: tuple, properties: list[str]) -> dict:
    """Creates a GeoJSON Point feature of a panda `Series`.
    The panda Series MUST contains axis `x0`, `y0` and properties listed by `properties` parameter.
    Series value of `properties` axis MUST be numbers.
    Computes X,Y coordinates of the feature point geometry in image coordinates reference system
    by implementing https://gdal.org/en/latest/tutorials/geotransforms_tut.html
    To do so, it uses the given `geo_transform` that should be retrieve from the source image
    by using gdal `GetGeoTransform` function of `DataSet` object.
    Series values of `properties` axis are put in the feature properties object.

    Args:
        series (Series): x0;y0;dx;dy;score series
        geo_transform (tuple): target image geotransform
        properties (list(str)): names of series axis to put as properties in the feature

    Returns:
        dict: GeoJSON feature
    """
    # Compute x, y coordinates in image CRS defined by geo_transform
    x = geo_transform[0] + series["x0"] * geo_transform[1] + series["y0"] * geo_transform[2]
    y = geo_transform[3] + series["x0"] * geo_transform[4] + series["y0"] * geo_transform[5]

    # Build and return the feature
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, y]},
        # 1* allow convert np.float32 to float that
        # allow json serialization (serialize np.float32 fail)
        # nan set to none for good json serialisation
        "properties": {
            prop: None if np.isnan(series[prop]) else 1 * series[prop] for prop in properties
        },
    }


class ProductGenerator:
    """Class to generate output products.
    Products are raster and vector files of Karios KP.
    """

    def __init__(
        self,
        config: RuntimeConfiguration,
        points: DataFrame,
        reference_image: GdalRasterImage,
    ):
        self._config: RuntimeConfiguration = config
        self._points: DataFrame = points
        self._reference_image: GdalRasterImage = reference_image

    def generate_products(self):
        """Generates:
        - mask if `gen_kp_mask` (-kpm)
        - KP Raster if `gen_delta_raster (-gip)`
        - KP geojson if reference image have projection

        Returns:
            list(str): list of generated product paths
        """
        product_paths = []
        if self._config.gen_kp_mask:
            product_paths.append(str(self._create_mask()))

        if self._config.gen_delta_raster:
            product_paths.append(str(self._create_intermediate_raster()))

        # always generate JSON if inputs products is geo referenced
        if not self._reference_image.get_epsg():
            logger.warning("Unable to generate KP GeoJSON, reference image not geo referenced")
        else:
            product_paths.append(str(self._create_kp_geojson()))

        return product_paths

    def _open_output_dataset(self, file_path: str, n_bands: int, e_type: int) -> gdal.Dataset:
        """Create an output GeoTIFF dataset with the same grid as the reference image."""
        ref = self._reference_image
        dataset = gdal.GetDriverByName("GTiff").Create(
            file_path,
            xsize=ref.x_size,
            ysize=ref.y_size,
            bands=n_bands,
            eType=e_type,
            options=["COMPRESS=LZW"],
        )
        if ref.projection:
            dataset.SetProjection(ref.projection)
        dataset.SetGeoTransform((ref.x_min, ref.x_res, 0, ref.y_max, 0, ref.y_res))
        return dataset

    def _create_intermediate_raster(self):
        logger.info("Create KP raster product")

        x_index = self._points["x0"].to_numpy().astype(int)
        y_index = self._points["y0"].to_numpy().astype(int)
        dx_vals = self._points["dx"].to_numpy().astype(np.float32)
        dy_vals = self._points["dy"].to_numpy().astype(np.float32)

        output_file_path = os.path.join(self._config.output_directory, "kp_delta.tif")
        dataset = self._open_output_dataset(output_file_path, 2, gdal.GDT_Float32)

        # Fill both bands with NaN at the GDAL level — no Python-side full-image array needed
        nan32 = float(np.float32(np.nan))
        for band_idx in (1, 2):
            b = dataset.GetRasterBand(band_idx)
            b.SetNoDataValue(nan32)
            b.Fill(nan32)
            b = None

        dx_band = dataset.GetRasterBand(1)
        dy_band = dataset.GetRasterBand(2)

        # Sort KPs by row so writes are sequential
        order = np.argsort(y_index, kind="stable")
        ys = y_index[order]
        xs = x_index[order]
        dxs = dx_vals[order]
        dys = dy_vals[order]

        # One reusable row buffer — O(x_size) memory regardless of image height
        row_buf = np.full((1, self._reference_image.x_size), nan32, dtype=np.float32)
        for row, start, end in _row_slices(ys):
            cols = xs[start:end]

            row_buf[0, cols] = dxs[start:end]
            dx_band.WriteArray(row_buf, 0, row)
            row_buf[0, cols] = nan32

            row_buf[0, cols] = dys[start:end]
            dy_band.WriteArray(row_buf, 0, row)
            row_buf[0, cols] = nan32

        dx_band = None
        dy_band = None
        dataset.FlushCache()
        dataset = None

        logger.info("KP raster product created")
        return output_file_path

    def _create_mask(self):
        logger.info("Create KP product mask")

        x_index = self._points["x0"].to_numpy().astype(int)
        y_index = self._points["y0"].to_numpy().astype(int)

        output_file_path = os.path.join(self._config.output_directory, "kp_mask.tif")
        dataset = self._open_output_dataset(output_file_path, 1, gdal.GDT_Byte)
        # GDT_Byte bands are zero-initialised by GDAL — no Fill() needed

        band = dataset.GetRasterBand(1)

        # Sort KPs by row so writes are sequential
        order = np.argsort(y_index, kind="stable")
        ys = y_index[order]
        xs = x_index[order]

        # One reusable row buffer — O(x_size) memory regardless of image height
        row_buf = np.zeros((1, self._reference_image.x_size), dtype=np.uint8)
        for row, start, end in _row_slices(ys):
            cols = xs[start:end]
            row_buf[0, cols] = 1
            band.WriteArray(row_buf, 0, row)
            row_buf[0, cols] = 0

        band = None
        dataset.FlushCache()
        dataset = None

        logger.info("KP product mask created")
        return output_file_path

    def _create_kp_geojson(self):
        logger.info("Create KP vector product")

        # configure properties to export in features
        columns_to_export = ["dx", "dy", "score", "radial error", "angle"]
        if "zncc_score" in self._points.columns:
            columns_to_export.append("zncc_score")

        # creates feature for each dataframe rows
        feature_as_series = self._points.apply(
            _to_feature,
            axis=1,
            geo_transform=self._reference_image.geo_transform,
            properties=columns_to_export,
        )

        feature_collection = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{self._reference_image.get_epsg()}"},
            },
            "features": feature_as_series.to_list(),
        }

        output_file = os.path.join(self._config.output_directory, "kp_delta.json")
        with open(output_file, "w", encoding="UTF8") as out:
            out.write(json.dumps(feature_collection, indent=3))

        logger.info("KP vector product created")

        return output_file
