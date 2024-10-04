# -*- coding: utf-8 -*-
# Copyright (c) 2024 Telespazio France.
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

from core.configuration import GlobalConfiguration
from core.image import GdalRasterImage

logger = logging.getLogger(__name__)


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
        "properties": {prop: 1 * series[prop] for prop in properties},
    }


class ProductGenerator:
    """Class to generate output products.
    Products are raster and vector files of Karios KP.
    """

    def __init__(
        self, config: GlobalConfiguration, points: DataFrame, reference_image: GdalRasterImage
    ):
        self._config: GlobalConfiguration = config
        self._points: DataFrame = points
        self._reference_image: GdalRasterImage = reference_image

    def generate_products(self):
        """Generates:
        - mask if `gen_kp_mask` (-kpm)
        - KP Raster if `gen_delta_raster (-gip)`
        - KP geojson if reference image have projection
        """
        if self._config.gen_kp_mask:
            self._create_mask()

        if self._config.gen_delta_raster:
            self._create_intermediate_raster()

        if not self._reference_image.get_epsg():
            logger.warning("Unable to generate KP GeoJSON, reference image not geo referenced")
        else:
            self._create_kp_geojson()

    def _create_intermediate_raster(self):
        logger.info("Create KP raster product")

        x_index = self._points["x0"].to_numpy().astype(int)
        y_index = self._points["y0"].to_numpy().astype(int)

        dx_band_array = np.full(
            [self._reference_image.y_size, self._reference_image.x_size], np.nan, dtype=float
        )
        dy_band_array = np.full(
            [self._reference_image.y_size, self._reference_image.x_size], np.nan, dtype=float
        )

        dx_band_array[y_index, x_index] = self._points["dx"]
        dy_band_array[y_index, x_index] = self._points["dy"]

        self._reference_image.to_raster(
            os.path.join(self._config.output_directory, "kp_delta.tif"),
            [dx_band_array, dy_band_array],
            gdal.GDT_Float32,
        )

        logger.info("KP raster product created")

    def _create_mask(self):
        logger.info("Create KP product mask")
        # Credits Jérôme
        x_index = self._points["x0"].to_numpy().astype(int)
        y_index = self._points["y0"].to_numpy().astype(int)
        final_mask = np.zeros(
            [self._reference_image.y_size, self._reference_image.x_size], dtype=np.uint8
        )
        final_mask[y_index, x_index] = 1
        self._reference_image.to_raster(
            os.path.join(self._config.output_directory, "kp_mask.tif"), final_mask
        )
        logger.info("KP product mask created")

    def _create_kp_geojson(self):
        logger.info("Create KP vector product")
        # creates feature for each dataframe rows
        feature_as_series = self._points.apply(
            _to_feature,
            axis=1,
            geo_transform=self._reference_image.geo_transform,
            properties=["dx", "dy", "score", "radial error", "angle"],
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
