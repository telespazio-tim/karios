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


"""Module for image classes."""
from __future__ import annotations

import logging
import os

from numpy.typing import NDArray
from osgeo import gdal, osr

logger = logging.getLogger()


def get_image_resolution(
    mon_img: GdalRasterImage,
    ref_img: GdalRasterImage,
    default_value: float | None = None,
) -> float:
    """Get the monitored image resolution.
    - If monitored image is not geo ref, it does not have a "real" pixel size.
        Use 'default_value' if provided, otherwise '1.0'
    - If monitored image have pixel size, and 'default_value' is set,
        ignore default and use monitored pixel size.

    Args:
        mon_img (GdalRasterImage): monitored image
        ref_img (GdalRasterImage): reference image, only for log information
        default_value (float | None, optional):
            Pixel size to consider for images that does not have a "real" pixel size.
            Defaults to None.

    Returns:
        float: the image resolution /pixel size to consider.
    """
    if not mon_img.have_pixel_resolution():
        if default_value is None:
            logger.warning(
                # pylint: disable-next=line-too-long
                "No pixel resolution provided by user and resolution cannot be read from image to monitor"
            )
        else:
            logger.info(
                "Input image does not have pixel size information, use provided pixel size %s",
                default_value,
            )
        return default_value

    if mon_img.have_pixel_resolution() and default_value is not None:
        logger.warning(
            # pylint: disable-next=line-too-long
            "User provides pixel resolution but resolution can be read from image to monitor, use input image resolution"
        )

    if abs(mon_img.x_res) != abs(mon_img.y_res):
        logger.warning(
            "Monitored image X pixel size and Y pixel size differ: %s, %s, consider X pixel size",
            mon_img.x_res,
            mon_img.y_res,
        )

    if ref_img.x_res and abs(mon_img.x_res) != abs(ref_img.x_res):
        logger.warning(
            "Monitored and reference images X resolution differ, %s vs %s",
            mon_img.x_res,
            ref_img.x_res,
        )
    if ref_img.x_res is None:
        logger.warning("Unable to get reference image pixel size")

    logger.info("Monitored image pixel size: %s", mon_img.x_res)
    return mon_img.x_res


class GdalRasterImage:
    # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Raster class using gdal to read image."""

    def __init__(self, filename):
        self.filepath = filename
        self.file_name = os.path.basename(self.filepath)
        self._read_header()
        self._array = None

    @property
    def x_res(self) -> int | float:
        """X resolution

        Returns:
            int | float: image X resolution
        """
        return self._geo[1]

    @property
    def y_res(self) -> int | float:
        """Y resolution

        Returns:
            int | float: image Y resolution, could be negative depending image SRS
        """
        return self._geo[5]

    @property
    def x_min(self) -> int | float:
        """UL pixel X coordinate

        Returns:
            int | float: UL X coordinate in image SRS if any
        """
        return self._geo[0]

    @property
    def y_max(self) -> int | float:
        """UL pixel Y coordinate

        Returns:
            int | float: UL Y coordinate in image SRS if any
        """
        return self._geo[3]

    def _read_header(self):
        # geo information
        dataset = gdal.Open(self.filepath)

        self._geo = dataset.GetGeoTransform()
        self.x_size = dataset.RasterXSize
        self.y_size = dataset.RasterYSize

        self.x_max = self.x_min + self.x_size * self.x_res
        self.y_min = self.y_max + self.y_size * self.y_res
        self.projection = dataset.GetProjection()

        dataset = None

    def have_pixel_resolution(self) -> bool:
        """Indicate if the image have a pixel size

        Returns:
            bool: 'True' if the image have a pixel size.
        """
        return bool(self.projection)

    def get_epsg(self) -> str | None:
        """Return image EPSG code as

        Returns:
            str|None: EPSG code as "EPSG: XXXX"
        """
        if self.have_pixel_resolution():
            srs = osr.SpatialReference(wkt=self.projection)
            return srs.GetAttrValue("PROJCS|AUTHORITY", 1)
        return None

    def read(self, band_id, x_off, y_off, x_size, y_size) -> NDArray:
        # pylint: disable=too-many-arguments
        """Read box of band denoted by 'band_id' at offset.
        This is a combination of gdal dataset 'GetRasterBand' and band 'ReadAsArray' usage.

        Args:
          band_id: band index, starting to 1
          x_off: image X_offset
          y_off: image Y offset
          x_size: X size of the box
          y_size: Y size of the box

        Returns:
          NDArray: extracted box in the image.
        """
        dst = gdal.Open(self.filepath)
        band = dst.GetRasterBand(band_id)
        data = band.ReadAsArray(x_off, y_off, x_size, y_size)
        dst = None
        return data

    def _get_array(self) -> NDArray:
        if self._array is None:
            dst = gdal.Open(self.filepath)
            band = dst.GetRasterBand(1)
            self._array = band.ReadAsArray()
            dst = None
        return self._array

    def to_raster(self, file_path: str, data: NDArray | list(NDArray), e_type=gdal.GDT_Byte):
        """Creates a raster file using input image size and res

        Args:
            file_path (str): destination mask file path
            data (NDArray | list(NDArray)): data to write, if is list, one array per band to write
            e_type (int): GDALDataType, see https://gdal.org/doxygen/gdal_8h.html#a22e22ce0a55036a96f652765793fb7a4
        """

        mono_band = not isinstance(data, list)

        nb_band = len(data) if not mono_band else 1

        creation_options = ["COMPRESS=LZW"]
        # for int8
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            file_path,
            xsize=self.x_size,
            ysize=self.y_size,
            bands=nb_band,
            eType=e_type,
            options=creation_options,
        )

        if self.projection:
            dataset.SetProjection(self.projection)

        geo_transform = (self.x_min, self.x_res, 0, self.y_max, 0, self.y_res)
        dataset.SetGeoTransform(geo_transform)

        if mono_band:
            dataset.GetRasterBand(1).WriteArray(data)
        else:
            for band_number, band_array in enumerate(data, 1):
                dataset.GetRasterBand(band_number).WriteArray(band_array)

        dataset.FlushCache()
        dataset = None

    def is_compatible_with(self, image: GdalRasterImage) -> bool:
        """Check that images have same geometric and geographic specifications

        Args:
            image (GdalRasterImage): image to compare with

        Returns:
            bool: True images have same geometric and geographic specifications
        """
        return (
            (self.projection == image.projection)
            and (self._geo == self._geo)
            and (self.x_size == image.x_size)
            and (self.y_size == image.y_size)
        )

    @property
    def image_information(self) -> str:
        """
        Returns:
            str: Images geometric and geographic specifications info
        """
        return f"""Projection: {self.projection}
        GetGeoTransform: {self._geo}
        X Size: {self.x_size}
        Y Size: {self.y_size}
        """

    array: NDArray = property(_get_array, doc="Access to image array (numpy array)")
