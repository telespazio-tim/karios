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

"""KTL module."""

import itertools
import logging
import os
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from skimage import io

from karios.core.configuration import KLTConfiguration
from karios.core.image import GdalRasterImage

logger = logging.getLogger(__name__)

LAPLACIAN_AUTO_CANDIDATES = [3, 5, 7, 9, 11]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8, no-op if already uint8."""
    if arr.dtype == np.uint8:
        return arr
    arr_min, arr_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    if arr_max > arr_min:
        return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


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
    ref_data: NDArray,
    image_data: NDArray,
    mask: NDArray,
    conf: KLTConfiguration,
    p0: NDArray | None = None,
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
        conf (KLTConfiguration): KLT configuration.
        p0 (NDArray | None): Optional pre-computed features to track.
            If None, they will be computed with `cv2.goodFeaturesToTrack`.

    Returns:
        tuple[DataFrame, int] | None: data frame of x, y, dx, dy, score
    """
    logger.info("Start tracking")

    if p0 is None:
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
        self,
        conf: KLTConfiguration,
        gen_laplacian: bool = False,
        out_dir: str | None = None,
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
        self._auto_selected_ksizes: list[tuple[int, int]] = []
        self._selected_polarities: list[str] = []

    def match(
        self,
        mon_img: GdalRasterImage,
        ref_img: GdalRasterImage,
        mask: GdalRasterImage | None,
    ) -> Iterator[DataFrame]:
        # pylint: disable=too-many-arguments
        """Run KLT on the image to monitor against a reference image and write result in csv file.

        Args:
            mon_img (GdalRasterImage): image to monitor
            ref_img (GdalRasterImage): reference image
            mask (GdalRasterImage | None): valid pixel mask of image to match

        Yields:
            Iterator[DataFrame]: dataframe generator
        """

        logger.info("KLT...")
        logger.info("%s %s", mon_img.x_size, mon_img.y_size)
        self._log_polarity_setting()

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

        self._log_polarity_summary()

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

        # mask_box = np.ones((ySize, xSize), np.uint8)
        # mask_box[img_box == 0] = 0
        # mask_box[ref_box == 0] = 0
        if mask:
            logger.info(
                "Read mask at offset x %s, y %s, with tile size %s, %s",
                x_off,
                y_off,
                x_size,
                y_size,
            )
            mask_box = mask.read(1, x_off, y_off, x_size, y_size)
        else:
            mask_box = (img_box != 0) & (ref_box != 0) & np.isfinite(ref_box) & np.isfinite(img_box)
            if mon_img.no_data_value is not None:
                mask_box &= img_box != mon_img.no_data_value
            if ref_img.no_data_value is not None:
                mask_box &= ref_box != ref_img.no_data_value
            mask_box = mask_box.astype(np.uint8)

        # check mask
        valid_pixels = len(mask_box[mask_box > 0])
        if valid_pixels == 0:
            logger.info("-- No valid pixels, skipping this tile")
            return None

        logger.info("Nb valid pixels: %s/%s", valid_pixels, x_size * y_size)

        # laplacian + tracking. `laplacian_invert_polarity` controls polarity:
        #   False  -> normal Laplacian (default)
        #   True   -> always invert monitored pixels before Laplacian
        #   "auto" -> run both and keep the higher-inlier-ratio result
        polarity_mode = self._conf.laplacian_invert_polarity
        if polarity_mode == "auto":
            normal_res, normal_dump = self._laplacian_track_once(
                img_box, ref_box, mask_box, invert_mon=False
            )
            inverted_res, inverted_dump = self._laplacian_track_once(
                img_box, ref_box, mask_box, invert_mon=True
            )
            results, dump = self._select_best_polarity(
                normal_res, normal_dump, inverted_res, inverted_dump
            )
        else:
            results, dump = self._laplacian_track_once(
                img_box, ref_box, mask_box, invert_mon=bool(polarity_mode)
            )

        if dump is not None:
            img_lap, ref_lap, mon_ksize, ref_ksize, invert_mon = dump
            if self._conf.laplacian_kernel_size == "auto":
                self._auto_selected_ksizes.append((mon_ksize, ref_ksize))
            if self._gen_laplacian:
                suffix = "_inv" if invert_mon else ""
                io.imsave(
                    os.path.join(
                        self._out_dir,
                        f"mon_laplacian{suffix}_k{mon_ksize}_{x_off}_{y_off}_{x_size}_{y_size}.tif",
                    ),
                    img_lap,
                )
                io.imsave(
                    os.path.join(
                        self._out_dir,
                        f"ref_laplacian_k{ref_ksize}_{x_off}_{y_off}_{x_size}_{y_size}.tif",
                    ),
                    ref_lap,
                )

        # clean large dataset
        ref_box = None
        img_box = None
        mask_box = None

        if not results:
            logger.warning(
                "No result for tile %s %s (%s %s)",
                x_off,
                y_off,
                mon_img.x_size,
                mon_img.y_size,
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

    @property
    def auto_selected_ksize(self) -> tuple[int, int] | None:
        """Return the most-common (mon_ksize, ref_ksize) pair chosen across all auto-mode tiles."""
        if not self._auto_selected_ksizes:
            return None
        return Counter(self._auto_selected_ksizes).most_common(1)[0][0]

    def _apply_laplacian_and_track(self, img_box, ref_box, mask_box, mon_ksize, ref_ksize):
        lap_img = cv2.Laplacian(_to_uint8(img_box), cv2.CV_8U, ksize=mon_ksize)
        lap_ref = cv2.Laplacian(_to_uint8(ref_box), cv2.CV_8U, ksize=ref_ksize)
        return klt_tracker(lap_ref, lap_img, mask_box, self._conf)

    def _log_polarity_setting(self) -> None:
        """Announce, at the start of a run, what polarity mode will be used."""
        mode = self._conf.laplacian_invert_polarity
        if mode == "auto":
            logger.info(
                "Laplacian polarity: 'auto' - each tile will run twice (normal and "
                "inverted monitored pixels), and the run with the higher inlier "
                "ratio will be kept"
            )
        elif mode:
            logger.info(
                "Laplacian polarity: 'inverted' - monitored pixels are inverted "
                "(255 - pixel) before Laplacian"
            )
        else:
            logger.info("Laplacian polarity: 'normal' - no inversion before Laplacian")

    def _log_polarity_summary(self) -> None:
        """When auto polarity was on, log the dominant choice across all tiles."""
        if self._conf.laplacian_invert_polarity != "auto":
            return
        if not self._selected_polarities:
            logger.info("Auto polarity: no tile produced a result")
            return
        counts = Counter(self._selected_polarities)
        total = sum(counts.values())
        dominant, dominant_count = counts.most_common(1)[0]
        details = ", ".join(f"{name}={n}/{total}" for name, n in counts.most_common())
        logger.info(
            "Auto polarity dominant choice: '%s' (%d/%d tiles, %s)",
            dominant,
            dominant_count,
            total,
            details,
        )

    @property
    def auto_selected_polarity(self) -> str | None:
        """Return the most-common polarity ('normal' or 'inverted') chosen across tiles
        when `laplacian_invert_polarity` is 'auto'."""
        if not self._selected_polarities:
            return None
        return Counter(self._selected_polarities).most_common(1)[0][0]

    def _laplacian_track_once(self, img_box, ref_box, mask_box, invert_mon: bool):
        """Run the Laplacian + KLT pipeline once.

        When invert_mon is True, the monitored image pixels are inverted
        (255 - uint8) before computing the Laplacian, which flips the feature
        polarity relative to the reference.

        Returns:
            tuple[result, dump]: result is the klt_tracker return value (or None);
                dump is (img_lap, ref_lap, mon_ksize, ref_ksize, invert_mon) for
                later optional writing, or None when no Laplacian was produced.
        """
        img_for_lap = (255 - _to_uint8(img_box)) if invert_mon else img_box

        ksize = self._conf.laplacian_kernel_size
        if ksize == "auto":
            result, _, best_ksize = self._match_tile_auto_ksize(img_for_lap, ref_box, mask_box)
            if best_ksize is None:
                return result, None
            mon_ksize, ref_ksize = best_ksize
            img_lap = cv2.Laplacian(_to_uint8(img_for_lap), cv2.CV_8U, ksize=mon_ksize)
            ref_lap = cv2.Laplacian(_to_uint8(ref_box), cv2.CV_8U, ksize=ref_ksize)
            return result, (img_lap, ref_lap, mon_ksize, ref_ksize, invert_mon)

        mon_ksize = ksize.get("mon", ksize.get("ref", 1)) if isinstance(ksize, dict) else ksize
        ref_ksize = ksize.get("ref", ksize.get("mon", 1)) if isinstance(ksize, dict) else ksize
        img_lap = cv2.Laplacian(_to_uint8(img_for_lap), cv2.CV_8U, ksize=mon_ksize)
        ref_lap = cv2.Laplacian(_to_uint8(ref_box), cv2.CV_8U, ksize=ref_ksize)
        result = klt_tracker(ref_lap, img_lap, mask_box, self._conf)
        return result, (img_lap, ref_lap, mon_ksize, ref_ksize, invert_mon)

    def _select_best_polarity(self, normal_res, normal_dump, inverted_res, inverted_dump):
        """Pick whichever polarity produced the higher inlier ratio.

        Records the winner's label so the dominant polarity can be reported at
        the end of the run. Returns (result, dump) of the winner.
        """
        candidates = []
        for label, res, dump in (
            ("normal", normal_res, normal_dump),
            ("inverted", inverted_res, inverted_dump),
        ):
            if res is None:
                continue
            points, ninit = res
            ratio = len(points) / ninit if ninit > 0 else 0.0
            candidates.append((label, ratio, res, dump))

        if not candidates:
            logger.info("Auto polarity: no candidate produced a result")
            return None, None

        candidates.sort(key=lambda c: c[1], reverse=True)
        label, ratio, result, dump = candidates[0]
        self._selected_polarities.append(label)
        logger.info("Auto polarity selected: %s (inlier ratio=%.3f)", label, ratio)
        return result, dump

    def _match_tile_auto_ksize(self, img_box, ref_box, mask_box):
        """Try all (mon_ksize, ref_ksize) combinations and return the result with the highest inlier ratio.

        Returns:
            tuple[tuple[DataFrame, int] | None, dict[tuple[int, int], float]]: best klt_tracker
                result and scores dict mapping each (mon_ksize, ref_ksize) pair to its inlier ratio.
        """
        combinations = list(itertools.product(LAPLACIAN_AUTO_CANDIDATES, repeat=2))

        # Pre-compute uint8 conversions once
        img_uint8 = _to_uint8(img_box)
        ref_uint8 = _to_uint8(ref_box)

        # Pre-compute Laplacians for each candidate kernel size
        mon_laplacians = {
            k: cv2.Laplacian(img_uint8, cv2.CV_8U, ksize=k) for k in LAPLACIAN_AUTO_CANDIDATES
        }
        ref_laplacians = {
            k: cv2.Laplacian(ref_uint8, cv2.CV_8U, ksize=k) for k in LAPLACIAN_AUTO_CANDIDATES
        }

        # Pre-compute features to track for each reference Laplacian
        feature_params = {
            "maxCorners": self._conf.maxCorners,
            "qualityLevel": self._conf.qualityLevel,
            "minDistance": self._conf.minDistance,
            "blockSize": self._conf.blocksize,
        }
        ref_p0s = {
            k: cv2.goodFeaturesToTrack(lap, mask=mask_box, **feature_params)
            for k, lap in ref_laplacians.items()
        }

        def _run(mon_ksize, ref_ksize):
            logger.info("Auto laplacian: trying mon_ksize=%s ref_ksize=%s", mon_ksize, ref_ksize)

            p0 = ref_p0s[ref_ksize]
            if p0 is None:
                logger.info(
                    "Auto laplacian: ref_ksize=%s -> no features extracted",
                    ref_ksize,
                )
                return (mon_ksize, ref_ksize), 0.0, None

            result = klt_tracker(
                ref_laplacians[ref_ksize],
                mon_laplacians[mon_ksize],
                mask_box,
                self._conf,
                p0=p0,
            )

            if result is None:
                logger.info("Auto laplacian: mon_ksize=%s ref_ksize=%s -> no result", mon_ksize, ref_ksize)
                return (mon_ksize, ref_ksize), 0.0, None
            points, ninit = result
            ratio = len(points) / ninit if ninit > 0 else 0.0
            logger.info("Auto laplacian: mon_ksize=%s ref_ksize=%s -> inlier ratio=%.3f (%d/%d)",
                        mon_ksize, ref_ksize, ratio, len(points), ninit)
            return (mon_ksize, ref_ksize), ratio, result

        with ThreadPoolExecutor() as executor:
            run_results = executor.map(lambda args: _run(*args), combinations)

        scores: dict[tuple[int, int], float] = {}
        best_result = None
        best_ratio = -1.0
        best_ksize = None

        for pair, ratio, result in run_results:
            scores[pair] = ratio
            if result is not None and ratio > best_ratio:
                best_ratio = ratio
                best_result = result
                best_ksize = pair

        logger.info("Auto laplacian selected: mon_ksize=%s ref_ksize=%s (inlier ratio=%.3f)",
                    best_ksize[0] if best_ksize else None,
                    best_ksize[1] if best_ksize else None,
                    best_ratio)
        return best_result, scores, best_ksize
