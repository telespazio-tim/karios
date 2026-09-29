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

from __future__ import annotations

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


def laplacian_to_uint8(response: NDArray, percentile: float = 98.0, power: float = 1.0) -> NDArray:
    """Rescale a signed Laplacian response to uint8 with a percentile-normalized power law.

    cv2.Laplacian's own CV_8U output clips every negative response to 0 and
    saturates large positive ones at 255 - measured on real imagery, that
    collapses 95%+ of pixels to the two extremes. That near-binary shape is
    not pure loss, though: on real multi-sensor pairs it made matching more
    robust, by discarding each sensor's exact gain/contrast and keeping only
    "is there a strong edge here". The bugs were that negative responses were
    discarded rather than mapped symmetrically, and that where the "binary"
    threshold actually fell was an accident of kernel size and image dtype
    rather than a deliberate, tunable choice.

    This fixes both while keeping the same "how binary" trade-off explicit.
    The response is first normalized by a percentile of its magnitude - the
    same idea `to_uint8` already applies to the raw image - so a few extreme
    pixels don't set the scale for the whole tile, and the sign survives
    (clip to [-1, 1] rather than [0, 1]). `power` then reshapes that
    normalized, sign-preserved value with a signed power law - `sign(y) *
    |y|**exponent`, where `exponent = 1 - power`:

    - `power=0` (`exponent=1`): the shape is left alone - a plain,
      percentile-normalized linear value, the "original" (non-binarized)
      signal.
    - `power=1` (`exponent=0`): every nonzero-response pixel saturates to
      +-1 - functionally the old near-binary behaviour, but symmetric and
      deliberate rather than an artifact of CV_8U's clipping.
    - values in between smoothly interpolate: lower exponents push weaker
      edges toward saturation sooner, without discarding the ones that
      remain below the noise floor (a response of exactly 0 always maps to
      128, regardless of `power`, since `sign(0) == 0`).

    Args:
        response: signed Laplacian response, any real dtype.
        percentile: percentile of the response magnitude used to normalize
            it before shaping. Defaults to 98.
        power: 0 keeps the percentile-normalized value as a plain linear
            signal; 1 pushes every nonzero response to full saturation
            (binary); in between interpolates via a signed power law.
            Defaults to 1.

    Returns:
        NDArray: uint8 array of the same shape, 128 where the response is
            exactly zero.
    """
    abs_response = np.abs(response)
    bound = np.percentile(abs_response, percentile)
    if bound <= 0:
        # Fewer than `100 - percentile`% of pixels carry any response at all -
        # common on tiles that are mostly flat with only a small, sharp
        # feature (a handful of edge pixels among a uniform background).
        # Falling back to the actual maximum still normalizes into (-1, 1]
        # instead of discarding those real edges as if the tile had none.
        bound = abs_response.max()
    if bound <= 0:
        return np.full(response.shape, 128, dtype=np.uint8)

    # The only clip left before shaping is the unavoidable one: fitting into
    # 8 bits eventually. `bound` calibrates contrast, not survival - it is
    # reached only by the top few percent, not most of the tile.
    y_norm = np.clip(response.astype(np.float64) / bound, -1.0, 1.0)

    exponent = 1.0 - power
    if exponent <= 0:
        shaped = np.sign(y_norm)
    else:
        shaped = np.sign(y_norm) * np.abs(y_norm) ** exponent

    return np.clip(np.round(shaped * 127.0 + 128.0), 0, 255).astype(np.uint8)
