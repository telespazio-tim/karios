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

"""Radiometric stretch used to prepare rasters for the 8-bit only OpenCV calls.

OpenCV's corner detection and optical flow want 8-bit input, so every matcher
has to map its pixels onto 0-255 first. How that mapping is chosen matters more
than it looks: a stretch driven by the array's minimum and maximum is decided
entirely by its two most extreme pixels, so one outlier or one infinity squeezes
the whole scene into a handful of DN values and the Laplacian has nothing left
to find. Measured on a Landsat-8 / Sentinel-2 pair, a single stray pixel took
matching from 3983 key points to none.

Float rasters carry such pixels routinely, where integer sensor products
generally do not, which is why this reads as a "float problem" even though the
stretch is what is actually fragile. Taking the bounds from percentiles of the
finite pixels instead costs a little contrast and is indifferent to how extreme
the outliers are.
"""

import numpy as np
from numpy.typing import NDArray

# Stretch bounds: low and high percentile of the finite pixels.
DEFAULT_PERCENTILES = (2.0, 98.0)


def to_uint8(arr: NDArray, percentiles: tuple[float, float] = DEFAULT_PERCENTILES) -> NDArray:
    """Scale an array to uint8 from the percentiles of its finite pixels.

    Args:
        arr: pixel data, any dtype. Returned unchanged when already uint8.
        percentiles: (low, high) percentile pair bounding the stretch.

    Returns:
        NDArray: uint8 array of the same shape. All-zero when the array holds no
            finite pixel, or when the two percentiles coincide so there is no
            range to stretch.
    """
    if arr.dtype == np.uint8:
        return arr

    values = arr.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(arr.shape, np.uint8)

    # percentile, not nanpercentile: the subset is already free of NaN *and* of
    # the infinities nanpercentile would happily carry into the bounds.
    low, high = np.percentile(values[finite], percentiles)
    if high <= low:
        return np.zeros(arr.shape, np.uint8)

    return np.clip((values - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
