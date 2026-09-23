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

"""Coarse-to-fine KLT matching.

`cv2.calcOpticalFlowPyrLK` recurses its own pyramid with a window of the same
pixel size at every level, so at level L that window spans `winSize * 2**L` of
the original image. Near a data edge it hangs off the image, the coarse estimate
is meaningless, and it seeds every finer level, which cannot recover: matching
erodes inward from every edge as the pyramid deepens.

This module descends the pyramid explicitly instead. Each level runs with
`maxLevel=0`, so the window is only ever `winSize` wide at the current
resolution, and the estimate handed down is a *smoothed* field fitted to the
reliable points rather than each point's own result. A point near an edge
therefore inherits a sane starting guess from its neighbours, and only the
finest level's `winSize / 2` band is lost.
"""

import logging

import cv2
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from karios.core.configuration import KLTConfiguration
from karios.core.radiometry import to_uint8

logger = logging.getLogger(__name__)

# Empirical, from four scenes (SPOT5, Landsat8/S2, MSS terrain, Sentinel-2).
# Forward-backward error, in current level pixels, for a point to be trusted
# when fitting that level's displacement field.
_COARSE_ACCEPT_PX = 0.5

# RANSAC inlier threshold when fitting the affine displacement field.
_RANSAC_REPROJ_PX = 2.0

# Below this many reliable points a level cannot fit a field, so the estimate
# from the previous level is carried down unchanged.
_MIN_POINTS_FOR_FIT = 50

# Final forward-backward consistency limit, matching `klt.klt_tracker`.
# Tighter than the shared default: the matcher was validated across four scenes
# with these bounds.
_LEVEL_PERCENTILES = (0.5, 99.5)

_BACK_THRESHOLD = 0.1


def _pyramid_level(image: NDArray, scale: int, ksize: int) -> NDArray:
    """Downsample then Laplacian-filter, in that order.

    Filtering after downsampling is what makes the coarse levels informative;
    downsampling an already-filtered image is not equivalent, as the fine
    detail the Laplacian responds to is exactly what decimation removes.
    """
    if scale == 1:
        small = image
    else:
        small = cv2.resize(
            image,
            (image.shape[1] // scale, image.shape[0] // scale),
            interpolation=cv2.INTER_AREA,
        )

    # nanpercentile drops NaN but keeps infinities, which then decide the bounds
    return cv2.Laplacian(to_uint8(small, _LEVEL_PERCENTILES), cv2.CV_8U, ksize=ksize)


def _track_level(
    ref_level: NDArray,
    mon_level: NDArray,
    points: NDArray,
    guess: NDArray,
    win_size: int,
) -> tuple[NDArray, NDArray]:
    """One forward/backward LK pass at a single resolution.

    Both passes are seeded: the forward pass with the predicted position, the
    backward pass with the points themselves, since that is where it should
    land. Seeding the backward pass at the forward result instead would ask it
    to converge the whole displacement from scratch, which fails at fine levels
    exactly where the displacement is largest in pixels.
    """
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
    lk_params = {
        "winSize": (win_size, win_size),
        "maxLevel": 0,
        "criteria": criteria,
        "flags": cv2.OPTFLOW_USE_INITIAL_FLOW,
    }

    forward, _, _ = cv2.calcOpticalFlowPyrLK(
        ref_level, mon_level, points, guess.copy(), **lk_params
    )
    backward, _, _ = cv2.calcOpticalFlowPyrLK(
        mon_level, ref_level, forward, points.copy(), **lk_params
    )

    return forward, abs(points - backward).reshape(-1, 2).max(-1)


def _fit_displacement_field(
    points: NDArray, displacements: NDArray, reliable: NDArray
) -> NDArray | None:
    """Fit an affine displacement field to the reliable points.

    Returns the field evaluated at every point, so unreliable ones are replaced
    by the model rather than kept. A local (per grid cell) field was measured as
    worse: cells holding few reliable points emit noisy medians that mislead the
    next level.
    """
    source = points[reliable].reshape(-1, 2).astype(np.float32)
    target = (points[reliable].reshape(-1, 2) + displacements[reliable]).astype(np.float32)

    transform, _ = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=_RANSAC_REPROJ_PX
    )
    if transform is None:
        return None

    predicted = (points.reshape(-1, 2) @ transform[:, :2].T) + transform[:, 2]

    return (predicted - points.reshape(-1, 2)).astype(np.float32)


def coarse_to_fine_tracker(
    ref_data: NDArray,
    image_data: NDArray,
    mask: NDArray,
    conf: KLTConfiguration,
    mon_ksize: int,
    ref_ksize: int,
    p0: NDArray | None = None,
) -> tuple[DataFrame, int] | None:
    """Match two images by descending a pyramid explicitly.

    Unlike `klt.klt_tracker` this takes the *raw* boxes, because each pyramid
    level is downsampled before being Laplacian-filtered.

    Args:
        ref_data (NDArray): reference image data, unfiltered
        image_data (NDArray): monitored image data, unfiltered
        mask (NDArray): region in which corners may be detected
        conf (KLTConfiguration): KLT configuration; `maxLevel` sets the coarsest
            level and `matching_winsize` the window at every level
        mon_ksize: Laplacian kernel size for the monitored image
        ref_ksize: Laplacian kernel size for the reference image
        p0 (NDArray | None): optional pre-computed features to track

    Returns:
        tuple[DataFrame, int] | None: frame of x0, y0, dx, dy, score and the
            initial key point count, or None when nothing could be matched.
    """
    logger.info("Start coarse-to-fine tracking, %s levels", conf.maxLevel)

    ref_full = _pyramid_level(ref_data, 1, ref_ksize)
    if p0 is None:
        p0 = cv2.goodFeaturesToTrack(
            ref_full,
            mask=mask,
            maxCorners=conf.maxCorners,
            qualityLevel=conf.qualityLevel,
            minDistance=conf.minDistance,
            blockSize=conf.blocksize,
        )

    if p0 is None:
        logger.info("No features extracted")
        return None

    initial_count = len(p0)
    flow = np.zeros((initial_count, 2), np.float32)

    for level in range(conf.maxLevel, -1, -1):
        scale = 2**level
        ref_level = _pyramid_level(ref_data, scale, ref_ksize)
        mon_level = _pyramid_level(image_data, scale, mon_ksize)

        points = (p0 / scale).astype(np.float32)
        guess = ((p0 + flow[:, None, :]) / scale).astype(np.float32)
        inside = (
            (points[:, 0, 0] > 1)
            & (points[:, 0, 1] > 1)
            & (points[:, 0, 0] < ref_level.shape[1] - 2)
            & (points[:, 0, 1] < ref_level.shape[0] - 2)
        )
        if inside.sum() < _MIN_POINTS_FOR_FIT // 2:
            logger.debug("Level %s too small to track, skipping", level)
            continue

        tracked, error = _track_level(ref_level, mon_level, points, guess, conf.matching_winsize)
        displacements = (tracked - points).reshape(-1, 2) * scale

        if level == 0:
            keep = (error < _BACK_THRESHOLD) & inside
            break

        reliable = (error < _COARSE_ACCEPT_PX) & inside
        if reliable.sum() > _MIN_POINTS_FOR_FIT:
            fitted = _fit_displacement_field(p0, displacements, reliable)
            if fitted is not None:
                flow = fitted
                logger.debug(
                    "Level %s: %s reliable points, median |flow| %.2f px",
                    level,
                    int(reliable.sum()),
                    float(np.median(np.hypot(flow[:, 0], flow[:, 1]))),
                )
    else:
        # maxLevel was negative, or every level was skipped
        return None

    if not keep.any():
        logger.info("No point survived coarse-to-fine matching")
        return None

    x0 = p0[keep, 0, 0]
    y0 = p0[keep, 0, 1]
    data_frame = DataFrame.from_dict(
        {
            "x0": x0,
            "y0": y0,
            "dx": displacements[keep, 0],
            "dy": displacements[keep, 1],
            "score": 1 - error[keep] / _BACK_THRESHOLD,
        }
    )

    logger.info("Coarse-to-fine tracking finished, %s points", len(data_frame))

    return data_frame, initial_count
