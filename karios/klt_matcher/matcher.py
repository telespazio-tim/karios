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

"""KTL module."""

import logging
import os
from collections.abc import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from skimage import io

from core.configuration import KLTConfiguration
from core.image import GdalRasterImage

logger = logging.getLogger()


def __filter_outliers(x0, y0, x1, y1, score):
    dx = x1 - x0
    dy = y1 - y0
    while True:
        ind = (
            (np.abs(dx - dx.mean()) < 3 * dx.std())
            & (np.abs(dy - dy.mean()) < 3 * dy.std())
            & (np.abs(dx - dx.mean()) < 20)
            & (np.abs(dy - dy.mean()) < 20)
        )
        if len(ind[ind == True]) == len(dx):  # pylint: disable=singleton-comparison
            break
        dx = dx[ind]
        dy = dy[ind]
        x0 = x0[ind]
        x1 = x1[ind]
        y0 = y0[ind]
        y1 = y1[ind]
        score = score[ind]
    return x0, y0, x1, y1, score


# """
# #Parameters :
# maxCorners=20000                        # Nombre total de KP au depart.
# matching_winsize=25                     # A remonter
# minDistance=10                          # Avoir 2 points a moins de 10 pixel
# blockSize=15                            # Pour la recherche des KPs - pas utiliser pour matcher.
# """


def klt_tracker(
    ref_data: NDArray, image_data: NDArray, mask: NDArray, conf: KLTConfiguration
) -> tuple[DataFrame, int] | None:
    """Run KLT.
    See :
    - https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    - https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

    Args:
        ref_data (NDArray): reference data for matching
        image_data (NDArray): data to match
        mask (NDArray): Optional region of interest.
            It specifies the region in which the corners are detected for `cv2.goodFeaturesToTrack`.
        max_corners (int, optional): Maximum number of corners.
            Used for matching with `cv2.goodFeaturesToTrack`. Defaults to 20000.
        matching_winsize (int, optional): Size of the search window at each pyramid level.
            Used by `cv2.calcOpticalFlowPyrLK` call. . Defaults to 25.
        outliers_filtering (bool, optional): apply outliers filtering. Defaults to False.

    Returns:
        tuple[DataFrame, int] | None: data frame of x, y, dx, dy, score
    """
    logger.info("Start tracking")
    # compute the initial point set
    # goodFeaturesToTrack input parameters
    feature_params = {
        "maxCorners": conf.maxCorners,
        "qualityLevel": conf.qualityLevel,
        "minDistance": conf.minDistance,
        "blockSize": conf.blocksize,
    }

    # goodFeaturesToTrack corner extraction-ShiThomasi Feature Detector
    p0 = cv2.goodFeaturesToTrack(ref_data, mask=mask, **feature_params)
    if p0 is None:
        logger.info("No features extracted")
        return None

    # define KLT parameters-for matching
    # info("Using window of size {} for matching.".format(matching_winsize))
    lk_params = {
        "winSize": (conf.matching_winsize, conf.matching_winsize),
        "maxLevel": 1,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
    }  # LSM input parameters - termination criteria for corner estimation/stopping criteria

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        ref_data, image_data, p0, None, **lk_params
    )  # LSM image matching- KLT tracker

    p0r, st, err = cv2.calcOpticalFlowPyrLK(
        image_data, ref_data, p1, None, **lk_params
    )  # LSM image matching- KLT tracker

    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    back_threshold = 0.1
    st = d < back_threshold

    # logger.info("Nb Bad Status: {} ".format(len(st[st == 0])))

    # filter with status
    st_valid = 1
    Ninit = len(p0)
    p0 = p0[st == st_valid]
    p1 = p1[st == st_valid]
    err = err[st == st_valid]
    d = d[st == st_valid]
    score = 1 - d / back_threshold
    x0 = p0[:, 0, 0].reshape(len(p0))
    y0 = p0[:, 0, 1].reshape(len(p0))
    x1 = p1[:, 0, 0].reshape(len(p1))
    y1 = p1[:, 0, 1].reshape(len(p1))

    if conf.outliers_filtering:
        logger.info("Filter outliers")
        x0, y0, x1, y1, score = __filter_outliers(x0, y0, x1, y1, score)

    # to dataframe
    data_frame = DataFrame.from_dict(
        {"x0": x0, "y0": y0, "dx": x1 - x0, "dy": y1 - y0, "score": score}
    )

    logger.info("Tracking finished")

    return data_frame, Ninit


class KLT:
    # pylint: disable=too-few-public-methods
    """Class to execute KLT matcher"""

    def __init__(
        self, conf: KLTConfiguration, gen_laplacian: bool = False, out_dir: str | None = None
    ):
        """Constructor

        Args:
            conf (KLTConfiguration): KLT configuration
            gen_laplacian: shall dump laplacian results
            out_dir (str | None, optional): laplacian result dir. Defaults to None.
        """
        self._conf: KLTConfiguration = conf
        self._gen_laplacian = gen_laplacian
        self._out_dir = out_dir

    def match(
        self,
        mon_img: GdalRasterImage,
        ref_img: GdalRasterImage,
        mask: GdalRasterImage,
    ) -> Iterator[DataFrame]:
        # pylint: disable=too-many-arguments
        """Run KLT on the image to monitor against a reference image and write result in csv file.

        Args:
            mon_img (GdalRasterImage): image to monitor
            ref_img (GdalRasterImage): reference image
            mask (GdalRasterImage): valid pixel mask of image to match

        Yields:
            Iterator[DataFrame]: dataframe generator
        """

        logger.info("KLT...")
        logger.info("%s %s", mon_img.x_size, mon_img.y_size)

        # iterate over N*N boxes : aim is to limit memory consumption.
        for x_off in range(0, mon_img.x_size, self._conf.tile_size):
            if x_off < self._conf.xStart:
                continue

            for y_off in range(0, mon_img.y_size, self._conf.tile_size):
                # run matcher on tile
                points = self._match_tile(x_off, y_off, mon_img, ref_img, mask)

                if points is None:
                    continue

                yield points

    def _match_tile(self, x_off, y_off, mon_img, ref_img, mask) -> DataFrame | None:
        logger.info("Tile: %s %s (%s %s)", x_off, y_off, mon_img.x_size, mon_img.y_size)

        # box size
        x_size = (
            self._conf.tile_size
            if x_off + self._conf.tile_size < mon_img.x_size
            else mon_img.x_size - x_off
        )
        y_size = (
            self._conf.tile_size
            if y_off + self._conf.tile_size < mon_img.y_size
            else mon_img.y_size - y_off
        )

        # read images
        ref_box = ref_img.read(1, x_off, y_off, x_size, y_size)
        img_box = mon_img.read(1, x_off, y_off, x_size, y_size)

        logger.info("mask...")
        # mask_box = np.ones((ySize, xSize), np.uint8)
        # mask_box[img_box == 0] = 0
        # mask_box[ref_box == 0] = 0
        if mask:
            mask_box = mask.read(1, x_off, y_off, x_size, y_size)
        else:
            mask_box = (img_box != 0) & (ref_box != 0) & np.isfinite(ref_box) & np.isfinite(img_box)
            mask_box = mask_box.astype(np.uint8)

        # check mask
        valid_pixels = len(mask_box[mask_box > 0])
        if valid_pixels == 0:
            logger.info("-- No valid pixels, skipping this tile")
            return None

        logger.info("Nb valid pixels: %s/%s", valid_pixels, x_size * y_size)

        # laplacian
        img_box = cv2.Laplacian(img_box, cv2.CV_8U, ksize=self._conf.laplacian_kernel_size)
        ref_box = cv2.Laplacian(ref_box, cv2.CV_8U, ksize=self._conf.laplacian_kernel_size)

        if self._gen_laplacian:
            io.imsave(
                os.path.join(self._out_dir, f"mon_laplacian_{x_off}_{y_off}_{x_size}_{y_size}.tif"),
                img_box,
            )
            io.imsave(
                os.path.join(self._out_dir, f"ref_laplacian_{x_off}_{y_off}_{x_size}_{y_size}.tif"),
                ref_box,
            )

        results = klt_tracker(
            ref_box,
            img_box,
            mask_box,
            self._conf,
        )

        if not results:
            logger.warning(
                "No result for tile %s %s (%s %s)", x_off, y_off, mon_img.x_size, mon_img.y_size
            )
            return None

        points, initial_nb_points = results

        points["x0"] = points["x0"] + x_off
        points["y0"] = points["y0"] + y_off

        logger.info("NbPoints(init/final): %s / %s", initial_nb_points, len(points.dx))
        logger.info("DX/DY(KLT) MEAN: %s / %s", points.dx.mean(), points.dy.mean())
        logger.info("DX/DY(KLT) STD: %s / %s", points.dx.std(), points.dy.std())

        points.sort_values(by=["x0", "y0"], inplace=True)
        return points
