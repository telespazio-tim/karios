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
from karios.core.errors import ConfigurationError
from karios.core.image import GdalRasterImage
from karios.matcher.coarse_to_fine import coarse_to_fine_tracker

logger = logging.getLogger(__name__)

LAPLACIAN_AUTO_CANDIDATES = [3, 5, 7, 9, 11]


def _to_uint8_legacy_minmax(arr: np.ndarray) -> np.ndarray:
    """Scale to uint8 from the array's own minimum and maximum.

    Fragile: the mapping is decided by the two most extreme pixels, so a single
    outlier or infinity flattens everything else. Kept for integer rasters only,
    because changing their stretch shifts every existing result by a couple of
    percent and the end-to-end reference data is pinned to it.

    TODO: drop this once that reference data is regenerated deliberately, and
    let `karios.core.radiometry.to_uint8` handle every dtype.
    """
    if arr.dtype == np.uint8:
        return arr
    arr_min, arr_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    if arr_max > arr_min:
        return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def _tracking_margin(matching_winsize: int, max_level: int) -> int:
    """Context needed around a tile so LK windows stay fully supported.

    The window is `matching_winsize` wide at the finest pyramid level and covers
    twice as much of the original image per level above it.
    """
    return (matching_winsize // 2) * 2**max_level


def _mask_margin(mask_box: NDArray, pad_x: int, pad_y: int, x_size: int, y_size: int) -> NDArray:
    """Zero everything outside the tile itself, keeping the margin as context only."""
    masked = np.zeros_like(mask_box)
    masked[pad_y : pad_y + y_size, pad_x : pad_x + x_size] = mask_box[
        pad_y : pad_y + y_size, pad_x : pad_x + x_size
    ]
    return masked


def _valid_mask(
    img_box: NDArray,
    ref_box: NDArray,
    mon_no_data: float | None,
    ref_no_data: float | None,
    no_values: list[int] | None,
) -> NDArray:
    """Build the matching mask for a pair of boxes.

    Excludes zero, each raster's declared no-data value, and any DN listed in
    `no_values`. The last matters when a product declares one no-data value but
    is actually filled with another: those pixels would otherwise be tracked as
    if they were image content.

    Args:
        img_box: monitored image data
        ref_box: reference image data
        mon_no_data: monitored raster declared no-data value, if any
        ref_no_data: reference raster declared no-data value, if any
        no_values: DN values to exclude from both images

    Returns:
        NDArray: uint8 mask, non-zero where the pixel can be matched.
    """
    mask = (img_box != 0) & (ref_box != 0) & np.isfinite(ref_box) & np.isfinite(img_box)
    if mon_no_data is not None:
        mask &= img_box != mon_no_data
    if ref_no_data is not None:
        mask &= ref_box != ref_no_data
    if no_values:
        mask &= ~np.isin(img_box, no_values)
        mask &= ~np.isin(ref_box, no_values)

    return mask.astype(np.uint8)


def _read_with_margin(
    image, x_off: int, y_off: int, x_size: int, y_size: int, margin: int
) -> tuple[NDArray, int, int]:
    """Read a tile plus up to `margin` pixels of real neighbouring data.

    The margin is clipped to the raster, so no pixel is ever invented. Synthetic
    padding is deliberately not used: mirrored or replicated content does not
    move consistently between the reference and the monitored image, so an LK
    window overlapping it estimates a worse flow than a truncated window does.

    Args:
        image: raster to read, exposing `read` and `x_size` / `y_size`
        x_off: tile X offset in the raster
        y_off: tile Y offset in the raster
        x_size: tile width
        y_size: tile height
        margin: pixels of context wanted on each side

    Returns:
        tuple[NDArray, int, int]: the enlarged box, and the position
            (left, top) of the tile origin inside it.
    """
    read_x = max(0, x_off - margin)
    read_y = max(0, y_off - margin)
    read_x_end = min(image.x_size, x_off + x_size + margin)
    read_y_end = min(image.y_size, y_off + y_size + margin)

    box = image.read(1, read_x, read_y, read_x_end - read_x, read_y_end - read_y)

    return box, x_off - read_x, y_off - read_y


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
        "maxLevel": conf.maxLevel,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
    }  # LSM input parameters - termination criteria for corner estimation/stopping criteria

    logger.info(
        "Start KLT tracking, %s levels, %s window size",
        lk_params["maxLevel"],
        lk_params["winSize"],
    )

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
        no_values: list[int] | None = None,
        coarse_to_fine: bool = False,
    ):
        """Constructor

        Args:
            conf (KLTConfiguration): KLT configuration
            gen_laplacian: shall dump laplacian results
            out_dir (str | None, optional): laplacian result dir. Defaults to None.
            no_values (list[int] | None, optional): DN values to exclude from
                matching, for products filled with a value other than their
                declared no-data. Defaults to None.
            coarse_to_fine (bool, optional): descend the pyramid explicitly rather
                than letting OpenCV recurse its own. Defaults to False.

        Raises:
            ConfigurationError: if coarse_to_fine is combined with an "auto"
                Laplacian kernel size.
        """
        if coarse_to_fine and conf.laplacian_kernel_size == "auto":
            raise ConfigurationError(
                'laplacian_kernel_size "auto" is not supported with coarse-to-fine '
                "matching: the kernel search compares full resolution Laplacians and "
                "assumes single resolution matching."
            )

        self._conf: KLTConfiguration = conf
        self._gen_laplacian = gen_laplacian
        self._out_dir = out_dir
        self._no_values = no_values
        self._coarse_to_fine = coarse_to_fine
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

        # read images, with a margin of real neighbouring pixels so key points at
        # the tile edge keep full LK window support across the seam
        # The monitored image drives the geometry, as it does for the tile loop;
        # the reference and the mask are read over that same window so the three
        # arrays stay aligned.
        margin = _tracking_margin(self._conf.matching_winsize, self._conf.maxLevel)
        img_box, pad_x, pad_y = _read_with_margin(mon_img, x_off, y_off, x_size, y_size, margin)
        ref_box = ref_img.read(1, x_off - pad_x, y_off - pad_y, img_box.shape[1], img_box.shape[0])

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
            # same window as the images, so the three arrays stay aligned
            mask_box = mask.read(
                1, x_off - pad_x, y_off - pad_y, img_box.shape[1], img_box.shape[0]
            )
            if self._no_values:
                mask_box = mask_box & _valid_mask(img_box, ref_box, None, None, self._no_values)
        else:
            mask_box = _valid_mask(
                img_box,
                ref_box,
                mon_img.no_data_value,
                ref_img.no_data_value,
                self._no_values,
            )

        # The margin is context for the LK windows only: forbid feature detection
        # there so every key point belongs to exactly one tile.
        mask_box = _mask_margin(mask_box, pad_x, pad_y, x_size, y_size)

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
                # drop the margin so the dump matches the geometry in its file name
                tile = (slice(pad_y, pad_y + y_size), slice(pad_x, pad_x + x_size))
                io.imsave(
                    os.path.join(
                        self._out_dir,
                        f"mon_laplacian{suffix}_k{mon_ksize}_{x_off}_{y_off}_{x_size}_{y_size}.tif",
                    ),
                    img_lap[tile],
                )
                io.imsave(
                    os.path.join(
                        self._out_dir,
                        f"ref_laplacian_k{ref_ksize}_{x_off}_{y_off}_{x_size}_{y_size}.tif",
                    ),
                    ref_lap[tile],
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

        # coordinates are relative to the margin-enlarged box, not the tile
        points["x0"] = points["x0"] - pad_x + x_off
        points["y0"] = points["y0"] - pad_y + y_off

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
        lap_img = cv2.Laplacian(_stretch(img_box), cv2.CV_8U, ksize=mon_ksize)
        lap_ref = cv2.Laplacian(_stretch(ref_box), cv2.CV_8U, ksize=ref_ksize)
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
        img_for_lap = (255 - _stretch(img_box)) if invert_mon else img_box

        ksize = self._conf.laplacian_kernel_size
        if ksize == "auto":
            result, _, best_ksize, best_laplacians = self._match_tile_auto_ksize(
                img_for_lap, ref_box, mask_box
            )
            if best_ksize is None:
                return result, None
            mon_ksize, ref_ksize = best_ksize
            # the search already filtered the tile at every candidate size
            img_lap, ref_lap = best_laplacians
            return result, (img_lap, ref_lap, mon_ksize, ref_ksize, invert_mon)

        mon_ksize = ksize.get("mon", ksize.get("ref", 1)) if isinstance(ksize, dict) else ksize
        ref_ksize = ksize.get("ref", ksize.get("mon", 1)) if isinstance(ksize, dict) else ksize
        img_lap = cv2.Laplacian(_to_uint8(img_for_lap), cv2.CV_8U, ksize=mon_ksize)
        ref_lap = cv2.Laplacian(_to_uint8(ref_box), cv2.CV_8U, ksize=ref_ksize)
        if self._coarse_to_fine:
            # takes the raw boxes: each level is downsampled before filtering
            result = coarse_to_fine_tracker(
                ref_box, img_for_lap, mask_box, self._conf, mon_ksize, ref_ksize
            )
        else:
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
            tuple: best klt_tracker result, scores dict mapping each
                (mon_ksize, ref_ksize) pair to its inlier ratio, the winning pair,
                and the winning pair's already-computed (mon, ref) Laplacians.
        """
        combinations = list(itertools.product(LAPLACIAN_AUTO_CANDIDATES, repeat=2))

        # Pre-compute uint8 conversions once
        img_uint8 = _stretch(img_box)
        ref_uint8 = _stretch(ref_box)

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
                logger.info(
                    "Auto laplacian: mon_ksize=%s ref_ksize=%s -> no result", mon_ksize, ref_ksize
                )
                return (mon_ksize, ref_ksize), 0.0, None
            points, ninit = result
            ratio = len(points) / ninit if ninit > 0 else 0.0
            logger.info(
                "Auto laplacian: mon_ksize=%s ref_ksize=%s -> inlier ratio=%.3f (%d/%d)",
                mon_ksize,
                ref_ksize,
                ratio,
                len(points),
                ninit,
            )
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

        logger.info(
            "Auto laplacian selected: mon_ksize=%s ref_ksize=%s (inlier ratio=%.3f)",
            best_ksize[0] if best_ksize else None,
            best_ksize[1] if best_ksize else None,
            best_ratio,
        )
        best_laplacians = (
            (mon_laplacians[best_ksize[0]], ref_laplacians[best_ksize[1]]) if best_ksize else None
        )

        return best_result, scores, best_ksize, best_laplacians
