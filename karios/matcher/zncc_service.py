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
This module contains class service for ZNCC computation
"""
import logging

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from karios.core.image import GdalRasterImage

logger = logging.getLogger(__name__)


def _zncc(img1, img2, u1, v1, u2, v2, n):
    """Compute ZNCC between two image patches."""
    # Extract patches with bounds checking
    patch1 = img1[u1 - n : u1 + n + 1, v1 - n : v1 + n + 1]
    patch2 = img2[u2 - n : u2 + n + 1, v2 - n : v2 + n + 1]

    # Normalize patches
    patch1_norm = (patch1 - np.mean(patch1)) / np.std(patch1)
    patch2_norm = (patch2 - np.mean(patch2)) / np.std(patch2)

    return np.mean(patch1_norm * patch2_norm)


def _zncc2(img1: NDArray, img2: NDArray, u1: int, v1: int, u2: int, v2: int, n: int) -> float:
    """Compute Zero-mean Normalized Cross-Correlation (ZNCC) between two image patches.

    ZNCC is a similarity measure that computes the correlation between two patches
    after normalizing them to have zero mean and unit variance. The result ranges
    from -1 (completely anti-correlated) to +1 (perfectly correlated), with 0
    indicating no correlation.

    The ZNCC is computed as:
        ZNCC = (1/N) * Σ[(I₁(i,j) - μ₁)(I₂(i,j) - μ₂)] / (σ₁ * σ₂)

    where N is the number of pixels in the patch, μ is the mean intensity,
    and σ is the standard deviation of each patch.

    Args:
        img1 (NDArray): First input image as a 2D array of floating-point values.
        img2 (NDArray): Second input image as a 2D array of floating-point values.
        u1 (int): Row coordinate of the center pixel in img1 for patch extraction.
        v1 (int): Column coordinate of the center pixel in img1 for patch extraction.
        u2 (int): Row coordinate of the center pixel in img2 for patch extraction.
        v2 (int): Column coordinate of the center pixel in img2 for patch extraction.
        n (int): Half-size of the square patch window. The actual patch size will be
            (2*n+1) x (2*n+1) pixels.

    Returns:
        float: ZNCC coefficient between the two patches. Values range from -1 to +1:
            - +1: Perfect positive correlation (identical patches)
            - 0: No correlation
            - -1: Perfect negative correlation (inverted patches)

    Raises:
        IndexError: If the patch window extends beyond the image boundaries.
        ValueError: If either patch has zero standard deviation (uniform intensity),
            making ZNCC undefined.

    Note:
        This function requires that the patch windows fit entirely within both images.
        For patches near image borders, consider padding the images or using smaller
        window sizes. Images should be of floating-point dtype for best numerical precision.

    References:
        Lewis, J.P. "Fast Normalized Cross-Correlation." Vision Interface, 1995.
        Briechle, K. and Hanebeck, U.D. "Template Matching using Fast Normalized
        Cross Correlation." Proceedings of SPIE, 2001.
    """

    if n < 0:
        raise ValueError("Window half-size n must be non-negative")

    # Bounds checking
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    if (
        u1 - n < 0
        or u1 + n >= h1
        or v1 - n < 0
        or v1 + n >= w1
        or u2 - n < 0
        or u2 + n >= h2
        or v2 - n < 0
        or v2 + n >= w2
    ):
        raise IndexError("Patch window extends beyond image boundaries")

    # Extract patches
    patch1 = img1[u1 - n : u1 + n + 1, v1 - n : v1 + n + 1]
    patch2 = img2[u2 - n : u2 + n + 1, v2 - n : v2 + n + 1]

    # Check for zero standard deviation
    std1: float = float(np.std(patch1))
    std2: float = float(np.std(patch2))

    if std1 == 0 or std2 == 0:
        raise ValueError("Cannot compute ZNCC: one or both patches have zero standard deviation")

    # Normalize patches
    patch1_norm = (patch1 - np.mean(patch1)) / std1
    patch2_norm = (patch2 - np.mean(patch2)) / std2

    return float(np.mean(patch1_norm * patch2_norm))


class ZNCCService:
    """Service class to compute Zero-mean Normalized Cross-Correlation (ZNCC) between two image patches"""

    def __init__(self):
        self._chip_size = 57
        # to avoid recompute it for each KP
        self._chip_margin = int((self._chip_size - 1) / 2)

    def compute_zncc(
        self, df: DataFrame, monitored: GdalRasterImage, reference: GdalRasterImage
    ) -> Series:
        """Compute ZNCC for each KP of the given dataframe.

        Args:
            df (DataFrame): dataframe with columns x0, y0, dx, dy
            monitored (GdalRasterImage): monitored image to extract patches
            reference (GdalRasterImage): reference image to extract patches

        Returns:
            Series: zncc score series, with same index as the given dataframe, contains NaN at index not computed
        """
        logger.info("Compute ZNCC for %s points", len(df))

        score = df.apply(self._compute_zncc, axis=1, monitored=monitored, reference=reference)

        monitored.clear_cache()
        reference.clear_cache()

        logger.info("ZNCC computation finish")

        return score

    def _compute_zncc(self, series: Series, monitored, reference):

        # refrerence KP coordinates
        # x0 and y0 are always float with .O
        x0 = int(series["x0"])
        y0 = int(series["y0"])
        x0_offset = x0 - self._chip_margin
        y0_offset = y0 - self._chip_margin

        # monitored KP coordinates
        # dx and dy are float, use nearest int value
        x1 = round(series["x0"] + series["dx"])
        y1 = round(series["y0"] + series["dy"])
        x1_offset = x1 - self._chip_margin
        y1_offset = y1 - self._chip_margin

        x0_max = reference.x_size - self._chip_margin
        y0_max = reference.y_size - self._chip_margin
        x1_max = monitored.x_size - self._chip_margin
        y1_max = monitored.y_size - self._chip_margin

        # verify top and left
        if x0_offset < 0 or y0_offset < 0 or x1_offset < 0 or y1_offset < 0:
            logger.warning("Point to close to image top or left boundaries, skip it")
            return np.nan

        # verify bottom and right
        if x0 > x0_max or y0 > y0_max or x1 > x1_max or y1 > y1_max:
            logger.warning("Point to close to image bottom or right boundaries, skip it")
            return np.nan

        chip_ref = self._extract_chip(x0, y0, reference)
        chip_mon = self._extract_chip(x1, y1, monitored)

        try:

            return _zncc2(
                chip_ref,
                chip_mon,
                28,
                28,
                28,
                28,
                21,
            )

        except ValueError as e:
            if "zero standard deviation" in str(e):
                logger.warning(
                    "Cannot compute ZNCC: one or both patches have zero standard deviation, returning NaN",
                    extra={"x0": int(series["x0"]), "y0": int(series["y0"])}
                )
                return np.nan
            else:
                # Re-raise other ValueErrors that are not related to zero standard deviation
                raise
        except IndexError as e:
            # Handle boundary errors gracefully by returning NaN
            if "beyond image boundaries" in str(e):
                logger.warning(
                    "Cannot compute ZNCC: patch extends beyond image boundaries, returning NaN",
                    extra={"x0": int(series["x0"]), "y0": int(series["y0"]), "error": str(e)}
                )
                return np.nan
            else:
                # Re-raise other IndexErrors that are not related to boundaries
                raise
        except Exception as e:
            logger.error(
                "Error while computing ZNCC",
                exc_info=e,
                stack_info=True,
            )
            return np.nan

    def _extract_chip(self, x: int, y: int, image: GdalRasterImage):

        # compute chip boundaries in image coordinates
        x_min = x - self._chip_margin
        x_max = x + self._chip_margin + 1
        y_min = y - self._chip_margin
        y_max = y + self._chip_margin + 1

        return image.array[y_min:y_max, x_min:x_max]
