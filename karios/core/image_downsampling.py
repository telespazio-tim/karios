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
"""Integer-ratio downsampling of GdalRasterImage rasters using block_reduce."""

import logging
from pathlib import Path

import numpy as np
from osgeo import gdal
from skimage.measure import block_reduce

from karios.core.image import GdalRasterImage

logger = logging.getLogger(__name__)


def downsample_raster(
    source: GdalRasterImage,
    ratio: int,
    out_path: Path,
) -> GdalRasterImage:
    """Write a downsampled copy of `source` to `out_path` and return it.

    Uses block_reduce with np.nanmean. Pixels equal to the raster's no-data
    sentinel are replaced with NaN before reduction so they are excluded from
    each block's average; blocks where every sub-pixel was no-data are
    restored to the sentinel afterwards.

    The output raster keeps the source origin; its pixel size is multiplied
    by `ratio`. Trailing zero padding introduced by block_reduce when the
    source dimensions aren't multiples of the ratio is dropped so the output
    is exactly floor(size/ratio) wide and tall.
    """
    if ratio < 1:
        raise ValueError(f"Downsampling ratio must be >= 1, got {ratio}")
    if ratio == 1:
        return source

    arr = source.array.astype(np.float64, copy=False)
    nodata = source.no_data_value
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    reduced = block_reduce(arr, block_size=(ratio, ratio), func=np.nanmean)

    if nodata is not None:
        reduced = np.where(np.isnan(reduced), nodata, reduced)

    new_y = source.y_size // ratio
    new_x = source.x_size // ratio
    reduced = reduced[:new_y, :new_x]

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(out_path),
        xsize=new_x,
        ysize=new_y,
        bands=1,
        eType=gdal.GDT_Float32,
        options=["COMPRESS=LZW"],
    )
    if source.projection:
        dataset.SetProjection(source.projection)
    dataset.SetGeoTransform(
        (source.x_min, source.x_res * ratio, 0, source.y_max, 0, source.y_res * ratio)
    )

    band = dataset.GetRasterBand(1)
    band.WriteArray(reduced)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    dataset.FlushCache()
    band = None
    dataset = None

    logger.info(
        "Downsampled %s by %dx: %dx%d -> %dx%d",
        source.file_name,
        ratio,
        source.x_size,
        source.y_size,
        new_x,
        new_y,
    )
    return GdalRasterImage(str(out_path))
