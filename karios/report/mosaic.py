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
"""Module to build a mosaic of the monitored and reference images.

Two modes make any misregistration visible:

- checkerboard: alternating tiles of the two images, where a shift breaks the
  features crossing the tile edges. The monitored image is tinted red and the
  reference image blue, so each tile tells its source at a glance.
- overlay: the monitored image in the red channel and the reference image in
  the green and blue channels. Where both agree the pixel is gray, a shift
  fringes the features in red on one side and cyan on the other.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, features

from karios.core.image import GdalRasterImage
from karios.report.overview_plot import build_invalid_mask

logger = logging.getLogger(__name__)

# Native resolution AVIF encoding quality and speed (0-10, 10 fastest): on a
# 10980 px scene, 90 keeps the mean error under 0.6 DN in 15 MB (60 MB as PNG),
# and speed 8 encodes in 14 s instead of 39 s at the default 6 for 15% larger files
MOSAIC_QUALITY = 90
MOSAIC_SPEED = 8
# Full resolution chroma: in the overlay, the difference between the two images
# lives in the chroma planes that the default 4:2:0 halves in both directions.
# On a 3660 px scene, 4:4:4 cuts its 99th percentile error from 27 to 11 DN for
# 18% larger files, at the same encoding time. The lightly tinted checkerboard
# is indifferent to it
MOSAIC_SUBSAMPLING = "4:4:4"
# Checkerboard tint, in OKLCh (the polar form of Oklab): each image takes its own
# hue in degrees at a light chroma, keeping the Oklab lightness of its gray level,
# so both tints look as bright as the gray and as each other. Oklab hues of the
# sRGB primaries: red 29, blue 264
MONITORED_HUE = 29  # red
REFERENCE_HUE = 264  # blue
MOSAIC_CHROMA = 0.02

# Oklab matrices, from https://bottosson.github.io/posts/oklab/
_LINEAR_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_OKLAB_TO_LMS = np.linalg.inv(_LMS_TO_OKLAB)
_LMS_TO_LINEAR_SRGB = np.linalg.inv(_LINEAR_SRGB_TO_LMS)


def _to_gray(img: GdalRasterImage, invalid: np.ndarray | None) -> np.ndarray:
    """Histogram equalize an image to uint8.

    The equalization maps each value to its rank among the finite, non-zero
    pixels not marked True in `invalid`, so both images spread evenly over the
    256 gray levels whatever their sensor and radiometric range. It runs on the
    raw values, before any 8 bit quantization, so no precision is lost to an
    intermediate stretch.

    Zero, non-finite and `invalid` pixels are set to 0 and take no part in the
    histogram: counting the fill value around a scene would spend a share of
    the gray levels on it.

    Args:
        img (GdalRasterImage): image to equalize
        invalid (np.ndarray|None): pixels to hide
    """
    array = img.array
    hidden = ~np.isfinite(array) | (array == 0)
    if invalid is not None:
        hidden |= invalid

    gray = np.zeros(array.shape, dtype=np.uint8)
    values = array[~hidden]
    if values.size == 0:
        return gray

    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts)
    if cdf[-1] > cdf[0]:
        # Classic equalization: the darkest value maps to 0, the brightest to 255
        lut = np.round((cdf - cdf[0]) / (cdf[-1] - cdf[0]) * 255).astype(np.uint8)
    else:
        # Constant image
        lut = np.zeros(unique.shape, dtype=np.uint8)
    gray[~hidden] = lut[inverse.ravel()]
    return gray


def checkerboard(shape: tuple[int, int], tile_size: int) -> np.ndarray:
    """Boolean checkerboard, True on the tiles whose (row + col) index is odd.

    Args:
        shape (tuple[int, int]): (rows, cols)
        tile_size (int): tile side in pixel

    Returns:
        np.ndarray: boolean array of `shape`
    """
    rows = np.arange(shape[0]) // tile_size
    cols = np.arange(shape[1]) // tile_size
    return (rows[:, np.newaxis] + cols[np.newaxis, :]) % 2 == 1


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB, in 0-1 on the last axis, to Oklab (L, a, b)."""
    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.cbrt(linear @ _LINEAR_SRGB_TO_LMS.T) @ _LMS_TO_OKLAB.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert Oklab (L, a, b) on the last axis to sRGB in 0-1, not clipped to the gamut."""
    linear = (np.asarray(lab, dtype=float) @ _OKLAB_TO_LMS.T) ** 3 @ _LMS_TO_LINEAR_SRGB.T
    magnitude = np.abs(linear)
    encoded = np.where(
        magnitude <= 0.0031308, 12.92 * magnitude, 1.055 * magnitude ** (1 / 2.4) - 0.055
    )
    return np.sign(linear) * encoded


def _tint(hue: float) -> np.ndarray:
    """Lookup table from gray level to RGB, tinted toward `hue` in OKLCh.

    Each gray level keeps its Oklab lightness and takes MOSAIC_CHROMA at `hue`,
    or less near black and white, where sRGB cannot show that much: the chroma
    is reduced to the gamut edge rather than the color clipped, which would
    shift its lightness.
    """
    lightness = srgb_to_oklab(np.repeat(np.arange(256)[:, np.newaxis] / 255, 3, axis=1))[:, 0]
    direction = np.array([np.cos(np.radians(hue)), np.sin(np.radians(hue))])

    def in_gamut(chroma: np.ndarray) -> np.ndarray:
        lab = np.column_stack([lightness, chroma[:, np.newaxis] * direction])
        rgb = oklab_to_srgb(lab)
        return ((rgb >= -1e-6) & (rgb <= 1 + 1e-6)).all(axis=1)

    # Bisect the largest in-gamut chroma up to MOSAIC_CHROMA, for every gray level at once
    low = np.zeros(256)
    high = np.full(256, float(MOSAIC_CHROMA))
    fits = in_gamut(high)
    low[fits] = high[fits]
    for _ in range(30):
        middle = (low + high) / 2
        fits = in_gamut(middle)
        low = np.where(fits, middle, low)
        high = np.where(fits, high, middle)

    rgb = oklab_to_srgb(np.column_stack([lightness, low[:, np.newaxis] * direction]))
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _tinted_checkerboard(
    mon_gray: np.ndarray, ref_gray: np.ndarray, mon_tiles: np.ndarray
) -> np.ndarray:
    """RGB frame with the monitored image in red on `mon_tiles`, the reference in blue elsewhere."""
    frame = _tint(REFERENCE_HUE)[ref_gray]
    frame[mon_tiles] = _tint(MONITORED_HUE)[mon_gray[mon_tiles]]
    return frame


def _overlay(red_gray: np.ndarray, cyan_gray: np.ndarray) -> np.ndarray:
    """RGB frame with `red_gray` in the red channel, `cyan_gray` in green and blue."""
    return np.stack([red_gray, cyan_gray, cyan_gray], axis=2)


def _write(mosaic: np.ndarray, output_stem: Path) -> Path:
    """Write `mosaic` at native resolution, in AVIF if available, lossless PNG otherwise.

    WebP is no fallback here, its images cannot exceed 16383 px on a side.
    """
    image = Image.fromarray(mosaic)
    if features.check("avif"):
        output_file = output_stem.with_name(f"{output_stem.name}.avif")
        image.save(
            output_file, quality=MOSAIC_QUALITY, speed=MOSAIC_SPEED, subsampling=MOSAIC_SUBSAMPLING
        )
    else:
        output_file = output_stem.with_name(f"{output_stem.name}.png")
        image.save(output_file, format="PNG")
    return output_file


def generate_mosaic(
    mon_image: GdalRasterImage,
    ref_image: GdalRasterImage,
    output_stem: Path,
    tile_size: int,
    mask: GdalRasterImage | None = None,
    no_values: list[float] | None = None,
    mode: str = "checkerboard",
) -> Path:
    """Write a color mosaic of both images.

    Each image is histogram equalized to 8 bit on its own, so both show the
    same contrast whatever their radiometry. The mask applies to the monitored
    image only. Hidden pixels are black in the image they belong to.

    In checkerboard mode, the top left tile shows the reference image, then
    tiles alternate with the monitored image, each tinted in its own color:
    monitored in red, reference in blue.

    In overlay mode, the monitored image fills the red channel and the reference
    image the green and blue ones, so aligned features are gray and shifted ones
    fringed in red and cyan.

    The mosaic is written at native resolution in `output_stem` with the
    `.avif` extension, `.png` if Pillow cannot encode AVIF.

    Args:
        mon_image (GdalRasterImage): monitored image
        ref_image (GdalRasterImage): reference image, same grid as the monitored one
        output_stem (Path): destination file path, without extension
        tile_size (int): tile side in pixel, checkerboard mode only
        mask (GdalRasterImage|None): optional mask applied to the monitored image.
            Pixels where mask == 0 are hidden.
        no_values (list[float]|None): optional list of DN values to hide in both images
        mode (str): "checkerboard" or "overlay"

    Returns:
        Path: mosaic file path
    """
    if mode not in ("checkerboard", "overlay"):
        raise ValueError(f"Unknown mosaic mode {mode!r}")
    shape = ref_image.array.shape
    if mode == "overlay":
        logger.info("Generating %sx%s overlay mosaic", shape[1], shape[0])
    else:
        logger.info("Generating %sx%s mosaic with %s px tiles", shape[1], shape[0], tile_size)

    mon_gray = _to_gray(mon_image, build_invalid_mask(mon_image, mask, no_values))
    ref_gray = _to_gray(ref_image, build_invalid_mask(ref_image, None, no_values))
    if mode == "overlay":
        mosaic = _overlay(mon_gray, ref_gray)
    else:
        mosaic = _tinted_checkerboard(mon_gray, ref_gray, checkerboard(shape, tile_size))

    return _write(mosaic, output_stem)
