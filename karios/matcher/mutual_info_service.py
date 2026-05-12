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
"""
This module contains class service for normalized mutual information computation
"""
import logging

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from karios.core.image import GdalRasterImage

logger = logging.getLogger(__name__)


def _mutual_info(patch1: NDArray, patch2: NDArray, bins: int = 32) -> float:
    """Compute normalized mutual information between two image patches.

    Uses Studholme's NMI formula: NMI = (H(X) + H(Y)) / H(X, Y), which ranges from 1
    (no shared information) to 2 (identical distributions).

    Args:
        patch1: First image patch as a 2D array.
        patch2: Second image patch as a 2D array.
        bins: Number of histogram bins for intensity discretisation.

    Returns:
        float: Normalized mutual information in [1, 2], or NaN if undefined.
    """
    hist_2d, _, _ = np.histogram2d(patch1.ravel(), patch2.ravel(), bins=bins)

    n = hist_2d.sum()
    if n == 0:
        return np.nan

    pxy = hist_2d / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))

    if hxy == 0:
        return np.nan

    return float((hx + hy) / hxy)


class MutualInfoService:
    """Service class to compute normalized mutual information between two image patches."""

    def __init__(self):
        self._chip_size = 57
        self._chip_margin = int((self._chip_size - 1) / 2)

    def compute_mutual_info(
        self, df: DataFrame, monitored: GdalRasterImage, reference: GdalRasterImage
    ) -> Series:
        """Compute normalized mutual information for each KP of the given dataframe.

        Args:
            df (DataFrame): dataframe with columns x0, y0, dx, dy
            monitored (GdalRasterImage): monitored image to extract patches
            reference (GdalRasterImage): reference image to extract patches

        Returns:
            Series: mutual info score series, with same index as df, contains NaN where not computed
        """
        logger.info("Compute mutual information for %s points", len(df))

        score = df.apply(
            self._compute_mutual_info, axis=1, monitored=monitored, reference=reference
        )

        monitored.clear_cache()
        reference.clear_cache()

        logger.info("Mutual information computation finish")

        return score

    def _compute_mutual_info(self, series: Series, monitored, reference):
        x0 = int(series["x0"])
        y0 = int(series["y0"])
        x0_offset = x0 - self._chip_margin
        y0_offset = y0 - self._chip_margin

        x1 = round(series["x0"] + series["dx"])
        y1 = round(series["y0"] + series["dy"])
        x1_offset = x1 - self._chip_margin
        y1_offset = y1 - self._chip_margin

        x0_max = reference.x_size - self._chip_margin
        y0_max = reference.y_size - self._chip_margin
        x1_max = monitored.x_size - self._chip_margin
        y1_max = monitored.y_size - self._chip_margin

        if x0_offset < 0 or y0_offset < 0 or x1_offset < 0 or y1_offset < 0:
            logger.warning("Point to close to image top or left boundaries, skip it")
            return np.nan

        if x0 > x0_max or y0 > y0_max or x1 > x1_max or y1 > y1_max:
            logger.warning("Point to close to image bottom or right boundaries, skip it")
            return np.nan

        chip_ref = self._extract_chip(x0, y0, reference)
        chip_mon = self._extract_chip(x1, y1, monitored)

        try:
            return _mutual_info(chip_ref, chip_mon)
        except Exception as e:
            logger.error("Error while computing mutual information", exc_info=e, stack_info=True)
            return np.nan

    def _extract_chip(self, x: int, y: int, image: GdalRasterImage):
        x_min = x - self._chip_margin
        x_max = x + self._chip_margin + 1
        y_min = y - self._chip_margin
        y_max = y + self._chip_margin + 1

        return image.array[y_min:y_max, x_min:x_max]
