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
"""Module to build a checkerboard mosaic of the monitored and reference images.

Alternating tiles of the two images make any misregistration visible as a
break of the features crossing the tile edges.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, features

from karios.core.image import GdalRasterImage
from karios.report.overview_plot import build_invalid_mask, display_range

logger = logging.getLogger(__name__)

# Native resolution AVIF encoding quality and speed (0-10, 10 fastest): on a
# 10980 px scene, 90 keeps the mean error under 0.6 DN in 15 MB (60 MB as PNG),
# and speed 8 encodes in 14 s instead of 39 s at the default 6 for 15% larger files
MOSAIC_QUALITY = 90
MOSAIC_SPEED = 8


def _to_gray(img: GdalRasterImage, invalid: np.ndarray | None) -> np.ndarray:
    """Stretch an image to uint8 with the overview plot contrast.

    Non-finite and `invalid` pixels are set to 0.

    Args:
        img (GdalRasterImage): image to stretch
        invalid (np.ndarray|None): pixels to hide
    """
    v_min, v_max = display_range(img, invalid)
    span = v_max - v_min
    array = img.array.astype(np.float32)
    if span > 0:
        scaled = (array - v_min) / span
    else:
        # Constant image, as matplotlib does with vmin == vmax
        scaled = np.zeros_like(array)
    gray = np.round(np.clip(np.nan_to_num(scaled), 0, 1) * 255).astype(np.uint8)

    hidden = ~np.isfinite(array)
    if invalid is not None:
        hidden |= invalid
    gray[hidden] = 0
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


def _write(mosaic: np.ndarray, output_stem: Path) -> Path:
    """Write `mosaic` at native resolution, in AVIF if available, lossless PNG otherwise.

    WebP is no fallback here, its images cannot exceed 16383 px on a side.
    """
    if features.check("avif"):
        output_file = output_stem.with_name(f"{output_stem.name}.avif")
        Image.fromarray(mosaic).save(output_file, quality=MOSAIC_QUALITY, speed=MOSAIC_SPEED)
    else:
        output_file = output_stem.with_name(f"{output_stem.name}.png")
        if not cv2.imwrite(str(output_file), mosaic):
            raise OSError(f"Cannot write mosaic {output_file}")
    return output_file


def generate_mosaic(
    mon_image: GdalRasterImage,
    ref_image: GdalRasterImage,
    output_stem: Path,
    tile_size: int,
    mask: GdalRasterImage | None = None,
    no_values: list[float] | None = None,
) -> Path:
    """Write a grayscale checkerboard mosaic of both images.

    The top left tile shows the reference image, then tiles alternate with the
    monitored image. Each image keeps the contrast stretch of the overview plot,
    so the mask is applied to the monitored image only. Hidden pixels are black.

    The mosaic is written at native resolution in `output_stem` with the
    `.avif` extension, `.png` if Pillow cannot encode AVIF.

    Args:
        mon_image (GdalRasterImage): monitored image
        ref_image (GdalRasterImage): reference image, same grid as the monitored one
        output_stem (Path): destination file path, without extension
        tile_size (int): tile side in pixel
        mask (GdalRasterImage|None): optional mask applied to the monitored image.
            Pixels where mask == 0 are hidden.
        no_values (list[float]|None): optional list of DN values to hide in both images

    Returns:
        Path: mosaic file path
    """
    shape = ref_image.array.shape
    logger.info("Generating %sx%s mosaic with %s px tiles", shape[1], shape[0], tile_size)

    mon_gray = _to_gray(mon_image, build_invalid_mask(mon_image, mask, no_values))
    mosaic = _to_gray(ref_image, build_invalid_mask(ref_image, None, no_values))
    mon_tiles = checkerboard(shape, tile_size)
    mosaic[mon_tiles] = mon_gray[mon_tiles]

    return _write(mosaic, output_stem)
