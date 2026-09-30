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
"""Module having class to create "products".
Products are (geo) raster or vector files created from KP and their properties (dx, dy, etc ...)
"""

import json
import logging
import os

import numpy as np
from osgeo import gdal
from pandas import DataFrame

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


def _to_features(points: DataFrame, geo_transform: tuple, properties: list[str]) -> list[dict]:
    """GeoJSON Point features of every row of `points`, located by `geo_transform`.

    Computed column by column, where a DataFrame.apply over the rows took 1-3 s
    for the 15k key points of a 30 m scene. The coordinates follow
    https://gdal.org/en/latest/tutorials/geotransforms_tut.html, rotation terms
    included; NaN properties become None (null), numbers Python floats.
    """
    x0 = points["x0"].to_numpy(dtype=np.float64)
    y0 = points["y0"].to_numpy(dtype=np.float64)
    xs = (geo_transform[0] + x0 * geo_transform[1] + y0 * geo_transform[2]).tolist()
    ys = (geo_transform[3] + x0 * geo_transform[4] + y0 * geo_transform[5]).tolist()
    values = points[properties].to_numpy(dtype=np.float64)
    rows = np.where(np.isnan(values), None, values).tolist()
    return [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": dict(zip(properties, row)),
        }
        for x, y, row in zip(xs, ys, rows)
    ]



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
        # The full geotransform: a rotated grid has non-zero terms 2 and 4
        dataset.SetGeoTransform(ref.geo_transform)
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
        if "mutual_info_score" in self._points.columns:
            columns_to_export.append("mutual_info_score")

        features = _to_features(
            self._points, self._reference_image.geo_transform, columns_to_export
        )

        feature_collection = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{self._reference_image.get_epsg()}"},
            },
            "features": features,
        }

        output_file = os.path.join(self._config.output_directory, "kp_delta.json")
        with open(output_file, "w", encoding="UTF8") as out:
            out.write(json.dumps(feature_collection, indent=3))

        logger.info("KP vector product created")

        return output_file
