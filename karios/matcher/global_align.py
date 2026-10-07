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
"""Global homography preprocessing via SIFT feature matching.

Pipeline:
    1. Preprocess both mon and ref to uint8 with CLAHE (equalizes radiometry
       between sensors).
    2. With a geotransform prior, work at a common resolution: the finer image
       is area-averaged down to the coarser one's pixel size and ref is cropped
       to mon's footprint plus a search margin. A mon several times finer than
       ref otherwise has keypoints of details ref cannot show, and a small mon
       footprint leaves most ref keypoints without any counterpart.
    3. With a prior, search the translation left by the georeferencing: zero-mean
       correlation of a fully valid central block of mon over the search window.
       ECC only corrects a few pixels, so a georeferencing kilometers off needs
       this coarse start.
    4. Detect SIFT keypoints + 128-dim float descriptors on both.
    5. Match with BFMatcher(NORM_L2), or a FLANN KD-tree for large keypoint
       sets, apply Lowe's ratio test + mutual (cross-check) filtering for
       robustness.
    6. Fit a 2D homography (8 DOF) using cv2.findHomography + RANSAC.
    7. Refine with cv2.findTransformECC(MOTION_HOMOGRAPHY) on Sobel gradient
       magnitudes (sensor-invariant), from every initial estimate: RANSAC, the
       prior and the translation search.
    8. With a prior, reject the refined estimates too far from it to be a
       georeferencing correction (reflection, scale, anisotropy, rotation or
       translation beyond the limits below), then keep the best gradient
       correlation, computed on the pixels all of them cover so the scores
       compare.
    9. Apply the resulting 3x3 homography to mon (and mask) via
       cv2.warpPerspective, rendered on mon's own grid, in mon's CRS, over its
       corrected footprint. Without a prior, rendered over mon's footprint in
       ref at a whole fraction of ref's pixel size, fine enough to keep mon's
       resolution.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from osgeo import gdal, osr

from karios.core.image import GdalRasterImage
from karios.core.radiometry import to_uint8

logger = logging.getLogger(__name__)

SIFT_NFEATURES = 10000  # default max number of keypoints kept per image, 0 = unlimited
SIFT_CONTRAST_THRESHOLD = 0.02  # default 0.04; lower → more keypoints in low-contrast regions
SIFT_EDGE_THRESHOLD = 10
# SIFT runs on tiles of images larger than this side, each read with a margin
# and keeping the keypoints of its core only. OpenCV's SIFT doubles the image
# before building its pyramid and holds about 200 bytes per input pixel: 6 GB
# for the 28 Mpx reference crop of a PhiSat scene on 10 m Sentinel-2. Its
# thresholds are absolute, so tiles find the keypoints the whole image would,
# except for the few larger than the margin next to a tile edge.
SIFT_TILE_PX = 2048
SIFT_TILE_MARGIN_PX = 128
LOWE_RATIO = 0.75
RANSAC_THRESHOLD_PX = 3.0
MIN_MATCHES = 4  # cv2.findHomography needs ≥4 point pairs; more = robuster
# Descriptor pairs above which matching uses a FLANN KD-tree instead of brute
# force. Brute force compares every pair, 36 min for the 62k x 484k keypoints of
# a PhiSat scene on a 10 m Sentinel-2 crop, and OpenCV's cannot even take more
# than 262143 descriptors to match against. The KD-tree search is approximate,
# which the Lowe ratio and cross-check filters absorb, and takes seconds.
BRUTE_FORCE_MAX_PAIRS = 100_000_000
FLANN_TREES = 5
FLANN_CHECKS = 64
ECC_MAX_ITERS = 200
# Iterations of the ECC probe run from every start; only the start whose probe
# correlates best is refined up to ECC_MAX_ITERS, again from that start. The
# other starts used to run their 200 iterations without converging, 51-69% of
# a 10 m PhiSat alignment, only to lose: after 25 iterations they already
# correlate ten times less. Restarting from the winner's probe instead would
# re-warp the image and drift to a neighbouring optimum: on PhiSat, 2.5 min for
# a CE90 1% worse than the probe's own start converging in 11 s.
ECC_PROBE_ITERS = 25
ECC_EPS = 1e-6
ECC_MARGIN_PX = 64  # ref kept around mon's pre-warped footprint for ECC to move into
MIN_VALID_PIXELS = 1000  # fewest valid pixels an ECC run or a correlation score needs

# Georeferencing correction limits, with a prior. The search window extends
# past mon's prior footprint by this share of its size on each side.
GEOREF_SEARCH_FRACTION = 0.5
MAX_SCALE_CHANGE = 1.5  # largest scale factor from the prior, either way
MAX_ANISOTROPY = 1.3  # largest ratio between the scale factors of both axes
MAX_ROTATION_DEG = 30.0
MIN_SHIFT_BLOCK_PX = 32  # smallest central block the translation search correlates
# The window is widened, doubling the margin up to the whole reference, when
# the best alignment lands this far into the margin on either axis, or
# correlates this poorly: the georeferencing may be off by more than the window
EDGE_OF_WINDOW = 0.75
LOW_GRADIENT_CORRELATION = 0.2
# Output pixel sizes within this share of mon's estimated one count as keeping
# its resolution: the scale estimate itself carries about a percent of noise
OUTPUT_RESOLUTION_TOLERANCE = 0.02


_NUMPY_TO_GDAL_DTYPE = {
    np.dtype("uint8"): gdal.GDT_Byte,
    np.dtype("int16"): gdal.GDT_Int16,
    np.dtype("uint16"): gdal.GDT_UInt16,
    np.dtype("int32"): gdal.GDT_Int32,
    np.dtype("uint32"): gdal.GDT_UInt32,
    np.dtype("float32"): gdal.GDT_Float32,
    np.dtype("float64"): gdal.GDT_Float64,
}


@dataclass
class GlobalAlignment:
    """Outcome of detect_global_alignment(): homography mon → ref."""

    matrix: np.ndarray  # 3x3 homography, mon pixel coords → ref pixel coords
    n_inliers: int
    n_matches: int
    # (name, 3x3 matrix, ECC score) for every refinement candidate that
    # converged and passed the plausibility check, including the one chosen
    # as `matrix`.
    candidates: list = field(default_factory=list)

    @property
    def score(self) -> float:
        """RANSAC inlier ratio in [0, 1]."""
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


class RasterWindows:
    """A raster's shape and pixels, read one window at a time rather than as a whole.

    `windows[y0:y1, x0:x1]` reads that window only; np.asarray(windows) reads
    the whole band. The alignment reads the reference around the monitored
    footprint: on a 10 m Sentinel-2 tile, 10 s and 625 MB for the whole
    JPEG 2000 against 4 s and 300 MB for a PhiSat crop.
    """

    def __init__(self, image: GdalRasterImage):
        self._image = image
        self.shape = (image.y_size, image.x_size)

    def __getitem__(self, key: tuple[slice, slice]) -> np.ndarray:
        rows, cols = key
        y0, y1, _ = rows.indices(self.shape[0])
        x0, x1, _ = cols.indices(self.shape[1])
        return self._image.read(1, x0, y0, x1 - x0, y1 - y0)

    def __array__(self, dtype=None):
        array = self._image.array
        return array if dtype is None else array.astype(dtype)


def _preprocess(arr: np.ndarray) -> np.ndarray:
    """uint8 stretch + CLAHE to equalize radiometry across the two images."""
    img = to_uint8(arr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _north_up(geo_transform: tuple) -> bool:
    return geo_transform[2] == 0 and geo_transform[4] == 0


def _prior_from_georefs(
    monitored: GdalRasterImage, reference: GdalRasterImage
) -> Optional[np.ndarray]:
    """Build the 3x3 homography mon_pixel → ref_pixel implied by the two georeferencings.

    Two north-up images in the same CRS give it exactly from their
    geotransforms. Otherwise a grid of mon pixels is mapped through mon's
    full geotransform, reprojected into ref's CRS and brought to ref pixels,
    and the homography is fitted to it: this takes any pair of CRS and rotated
    or mirrored grids, like a PhiSat scene in WGS 84 on a Sentinel-2 tile in
    UTM, fitted within a reference pixel over 22 km.

    Returns None when no usable prior can be built (an image without CRS, or
    a reprojection failing).
    """
    if not monitored.projection or not reference.projection:
        return None
    mon_geo, ref_geo = monitored.geo_transform, reference.geo_transform
    try:
        same_crs = bool(monitored.spatial_ref.IsSame(reference.spatial_ref))
    except Exception:
        return None
    if same_crs and _north_up(mon_geo) and _north_up(ref_geo):
        sx = monitored.x_res / reference.x_res
        sy = monitored.y_res / reference.y_res
        tx = (monitored.x_min - reference.x_min) / reference.x_res
        ty = (monitored.y_max - reference.y_max) / reference.y_res
        # Geotransforms place pixel corners, OpenCV pixel centers: see _pixel_scale()
        return np.array(
            [[sx, 0.0, tx + (sx - 1) / 2], [0.0, sy, ty + (sy - 1) / 2], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return _fit_prior(monitored, reference)


PRIOR_GRID = 11  # mon pixels per side reprojected to fit a prior across CRS


def _fit_prior(monitored: GdalRasterImage, reference: GdalRasterImage) -> Optional[np.ndarray]:
    """Least squares homography mon_pixel → ref_pixel on a reprojected grid of mon pixels."""
    mon_srs = osr.SpatialReference(wkt=monitored.projection)
    ref_srs = osr.SpatialReference(wkt=reference.projection)
    for srs in (mon_srs, ref_srs):
        # x = easting or longitude, whatever the CRS definition's axis order
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ref_inverse = gdal.InvGeoTransform(reference.geo_transform)
    if ref_inverse is None:
        return None

    # Pixel centers, on OpenCV's integers; the geotransform maps corners, half a pixel out
    cols, rows = np.meshgrid(
        np.linspace(0, monitored.x_size - 1, PRIOR_GRID),
        np.linspace(0, monitored.y_size - 1, PRIOR_GRID),
    )
    mon_px = np.column_stack([cols.ravel(), rows.ravel()])
    try:
        geo = [gdal.ApplyGeoTransform(monitored.geo_transform, x + 0.5, y + 0.5) for x, y in mon_px]
        reprojected = osr.CoordinateTransformation(mon_srs, ref_srs).TransformPoints(geo)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Cannot reproject mon into ref's CRS: %s", e)
        return None
    ref_px = np.array([gdal.ApplyGeoTransform(ref_inverse, x, y) for x, y, *_ in reprojected])
    ref_px -= 0.5
    if not np.isfinite(ref_px).all():
        return None

    matrix, _ = cv2.findHomography(mon_px, ref_px, 0)
    if matrix is None:
        return None
    residual = np.abs(_footprint_points(matrix, mon_px) - ref_px).max()
    logger.info(
        "Geotransform prior reprojected from mon's CRS: fitted within %.2f ref px", residual
    )
    return matrix


@dataclass
class _WorkFrame:
    """mon and a crop of ref at a common resolution, with the maps back to full resolution."""

    mon: np.ndarray  # uint8 mon at the working resolution
    mon_valid: np.ndarray  # bool, pixels of `mon` fully covered by valid data
    ref: np.ndarray  # uint8 crop of ref at the working resolution
    mon_to_work: np.ndarray  # 3x3, mon full resolution px -> `mon` px
    ref_to_work: np.ndarray  # 3x3, ref full resolution px -> `ref` crop px
    margin: float  # search margin around mon's prior footprint, in `ref` px
    covers_ref: bool  # the crop is the whole reference: no wider window exists

    def to_work(self, matrix: np.ndarray) -> np.ndarray:
        """mon → ref homography at full resolution, expressed between the working images."""
        return self.ref_to_work @ matrix @ np.linalg.inv(self.mon_to_work)

    def to_full(self, matrix: np.ndarray) -> np.ndarray:
        """Homography between the working images, expressed at full resolution."""
        return np.linalg.inv(self.ref_to_work) @ matrix @ self.mon_to_work


def _pixel_scale(sx: float, sy: float) -> np.ndarray:
    """3x3 map between pixel grids of one footprint, the second `sx` x `sy` times the first.

    OpenCV puts pixel centers on integer coordinates, so scaling a grid also
    moves them by half a pixel of the difference: without it an image
    downsampled 7x lands 0.43 of its pixel off.
    """
    return np.array([[sx, 0.0, (sx - 1) / 2], [0.0, sy, (sy - 1) / 2], [0.0, 0.0, 1.0]])


def _valid_pixels(arr: np.ndarray) -> np.ndarray:
    """Finite, non-zero pixels: zero is the fill value the alignment treats as no data."""
    return np.isfinite(arr) & (arr != 0)


def _footprint(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """Corners of a width x height image mapped by `matrix`, as a 4x2 array."""
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], float).T
    mapped = matrix @ corners
    return (mapped[:2] / mapped[2]).T


def _work_frame(
    mon_arr: np.ndarray, ref_arr, prior: np.ndarray, fraction: float = GEOREF_SEARCH_FRACTION
) -> Optional[_WorkFrame]:
    """Crop ref around mon's prior footprint, preprocess both, bring them to the coarser resolution.

    The crop extends `fraction` of the footprint's size past it on each side.
    `ref_arr` may be a RasterWindows, which reads that crop only. Only the crop
    of ref is preprocessed: a small mon on a 10 m tile would otherwise stretch
    and equalize 120 Mpx to use a few of them. Returns None when the prior
    footprint does not overlap ref.
    """
    # Pixel size of mon relative to ref, from the prior's area scale
    scale = float(np.sqrt(abs(np.linalg.det(prior[:2, :2]))))
    mh, mw = mon_arr.shape
    rh, rw = ref_arr.shape

    corners = _footprint(prior, mw, mh)
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    margin = fraction * max(x1 - x0, y1 - y0)
    cx0, cy0 = max(0, int(np.floor(x0 - margin))), max(0, int(np.floor(y0 - margin)))
    cx1, cy1 = min(rw, int(np.ceil(x1 + margin))), min(rh, int(np.ceil(y1 + margin)))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    crop = np.array([[1.0, 0.0, -cx0], [0.0, 1.0, -cy0], [0.0, 0.0, 1.0]])

    mon_u8, mon_valid = _preprocess(mon_arr), _valid_pixels(mon_arr)
    mon_work, valid_work = mon_u8, mon_valid
    if scale < 1:
        size = (max(1, round(mw * scale)), max(1, round(mh * scale)))
        mon_work = cv2.resize(mon_u8, size, interpolation=cv2.INTER_AREA)
        # A working pixel is valid only when every mon pixel under it is
        coverage = cv2.resize(mon_valid.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        valid_work = coverage > 0.999
    mon_to_work = _pixel_scale(mon_work.shape[1] / mw, mon_work.shape[0] / mh)

    ref_work = _preprocess(ref_arr[cy0:cy1, cx0:cx1])
    ch, cw = ref_work.shape
    if scale > 1:
        size = (max(1, round(cw / scale)), max(1, round(ch / scale)))
        ref_work = cv2.resize(ref_work, size, interpolation=cv2.INTER_AREA)
    ref_scale = _pixel_scale(ref_work.shape[1] / cw, ref_work.shape[0] / ch)

    # Straighten mon into ref's orientation when the prior rotates, mirrors or
    # shears it: SIFT is not mirror invariant, the translation search needs
    # both images north-up alike, and ECC only corrects small differences
    rectify = ref_scale @ crop @ prior @ np.linalg.inv(mon_to_work)
    rectify /= rectify[2, 2]
    aligned_axes = np.allclose(rectify[:2, :2], np.eye(2), atol=1e-3)
    if not aligned_axes or np.abs(rectify[2, :2]).max() > 1e-9:
        wh, ww = mon_work.shape
        corners = _footprint(rectify, ww, wh)
        (bx0, by0), (bx1, by1) = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
        to_canvas = np.array([[1.0, 0.0, -bx0], [0.0, 1.0, -by0], [0.0, 0.0, 1.0]]) @ rectify
        size = (int(bx1 - bx0), int(by1 - by0))
        m32 = to_canvas.astype(np.float32)
        mon_work = cv2.warpPerspective(mon_work, m32, size, flags=cv2.INTER_LINEAR)
        valid_u8 = valid_work.astype(np.uint8)
        covered = cv2.warpPerspective(valid_u8, m32, size, flags=cv2.INTER_NEAREST)
        # Bilinear warping blends the edge of the data with the zero fill
        valid_work = cv2.erode(covered, np.ones((3, 3), np.uint8)) > 0
        mon_to_work = to_canvas @ mon_to_work
        logger.info("mon straightened into ref's orientation by the prior: %s", _decompose(rectify))

    return _WorkFrame(
        mon=mon_work,
        mon_valid=valid_work,
        ref=ref_work,
        mon_to_work=mon_to_work,
        ref_to_work=ref_scale @ crop,
        margin=margin * ref_scale[0, 0],
        covers_ref=(cx0, cy0, cx1, cy1) == (0, 0, rw, rh),
    )


def _search_translation(frame: _WorkFrame, start: np.ndarray) -> Optional[np.ndarray]:
    """Correct the translation of `start` (working images homography) by template matching.

    Correlates the largest fully valid square block centered in mon with the
    whole ref crop, zero-mean normalized so the radiometry of each sensor does
    not matter. Assumes `start` keeps mon north-up at the working scale, as a
    geotransform prior between north-up images does. Returns None when mon has
    no valid central block large enough.
    """
    mh, mw = frame.mon.shape
    cx, cy = mw // 2, mh // 2
    if not frame.mon_valid[cy, cx]:
        return None
    half = 1
    while (
        half < min(cx, cy, mw - cx, mh - cy)
        and frame.mon_valid[cy - half - 1 : cy + half + 1, cx - half - 1 : cx + half + 1].all()
    ):
        half += 1
    if 2 * half < MIN_SHIFT_BLOCK_PX:
        logger.info("Translation search skipped: valid central block of %d px only", 2 * half)
        return None
    block = frame.mon[cy - half : cy + half, cx - half : cx + half]
    if block.shape[0] > frame.ref.shape[0] or block.shape[1] > frame.ref.shape[1]:
        return None

    scores = cv2.matchTemplate(frame.ref, block, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (bx, by) = cv2.minMaxLoc(scores)
    predicted = start @ np.array([cx - half, cy - half, 1.0])
    predicted = predicted[:2] / predicted[2]
    dx, dy = bx - predicted[0], by - predicted[1]
    logger.info(
        "Translation search: %dx%d px block, shift from prior=(%+.1f, %+.1f) working px  "
        "correlation=%.3f",
        2 * half,
        2 * half,
        dx,
        dy,
        peak,
    )
    shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return shift @ start


def _at_mon_center(matrix: np.ndarray, frame: _WorkFrame) -> tuple[np.ndarray, np.ndarray]:
    """Position in ref px and 2x2 Jacobian of the mon → ref `matrix` at mon's center."""
    mh, mw = frame.mon.shape
    center = np.linalg.inv(frame.mon_to_work) @ np.array([mw / 2, mh / 2, 1.0])
    p = matrix @ center
    w = p[2]
    jacobian = (matrix[:2, :2] * w - np.outer(p[:2], matrix[2, :2])) / w**2
    return p[:2] / w, jacobian


def _plausibility(matrix: np.ndarray, prior: np.ndarray, frame: _WorkFrame) -> Optional[str]:
    """Why `matrix` is too far from `prior` to be a georeferencing correction, None if it is not.

    Compares the linear parts at mon's center, where the homography is
    linearized, and the positions of mon's center.
    """
    pos, jacobian = _at_mon_center(matrix, frame)
    prior_pos, prior_jacobian = _at_mon_center(prior, frame)
    relative = jacobian @ np.linalg.inv(prior_jacobian)
    if np.linalg.det(relative) <= 0:
        return "mirrors the image"
    u, singular, vt = np.linalg.svd(relative)
    if singular[0] > MAX_SCALE_CHANGE or singular[1] < 1 / MAX_SCALE_CHANGE:
        return f"scales it by {singular[1]:.2f}-{singular[0]:.2f} (limit x{MAX_SCALE_CHANGE})"
    if singular[0] / singular[1] > MAX_ANISOTROPY:
        return f"anisotropy {singular[0] / singular[1]:.2f} (limit {MAX_ANISOTROPY})"
    rotation = u @ vt
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    if abs(angle) > MAX_ROTATION_DEG:
        return f"rotates it by {angle:+.1f}° (limit {MAX_ROTATION_DEG}°)"
    # Margin in ref full resolution px, from the working one
    margin = frame.margin / frame.ref_to_work[0, 0]
    distance = float(np.hypot(*(pos - prior_pos)))
    if distance > margin * np.sqrt(2):
        return f"moves its center by {distance:.0f} px (search window {margin:.0f} px)"
    return None


def _gradient_correlation(
    frame: _WorkFrame, matrices: list[np.ndarray]
) -> tuple[list[float], int]:
    """Zero-mean correlation of Sobel magnitudes between ref and mon warped by each matrix.

    Computed on the pixels every warped mon covers, so the scores compare.
    `matrices` map the working images. Returns the scores and the pixel count.
    """
    rh, rw = frame.ref.shape
    template = _sobel_magnitude(frame.ref)
    valid_u8 = frame.mon_valid.astype(np.uint8)
    warped, common = [], np.ones((rh, rw), dtype=bool)
    for matrix in matrices:
        m32 = matrix.astype(np.float32)
        warped.append(
            _sobel_magnitude(cv2.warpPerspective(frame.mon, m32, (rw, rh), flags=cv2.INTER_LINEAR))
        )
        covered = cv2.warpPerspective(valid_u8, m32, (rw, rh), flags=cv2.INTER_NEAREST)
        # Sobel reads a pixel's neighbours: keep away from the edge of the data
        common &= cv2.erode(covered, np.ones((5, 5), np.uint8)) > 0
    count = int(common.sum())
    if count < MIN_VALID_PIXELS:
        return [float("nan")] * len(matrices), count

    t = template[common] - template[common].mean()
    scores = []
    for image in warped:
        i = image[common] - image[common].mean()
        denominator = float(np.sqrt((t * t).sum() * (i * i).sum()))
        scores.append(float((t * i).sum() / denominator) if denominator > 0 else float("nan"))
    return scores, count


def _knn_match(query: np.ndarray, train: np.ndarray, k: int) -> list:
    """k nearest `train` descriptors of each `query` one, brute force or FLANN by size.

    See BRUTE_FORCE_MAX_PAIRS.
    """
    if len(query) * len(train) <= BRUTE_FORCE_MAX_PAIRS:
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(query, train, k=k)
    kdtree = 1  # FLANN_INDEX_KDTREE
    matcher = cv2.FlannBasedMatcher(
        {"algorithm": kdtree, "trees": FLANN_TREES}, {"checks": FLANN_CHECKS}
    )
    return matcher.knnMatch(query, train, k=k)


def _detect_sift(
    sift: cv2.SIFT, img: np.ndarray, nfeatures: int
) -> tuple[list, Optional[np.ndarray]]:
    """SIFT keypoints and descriptors of `img`, by tiles when it is large, see SIFT_TILE_PX.

    `sift` keeps at most `nfeatures` keypoints per tile (0: all), before
    computing their descriptors: a keypoint among the image's `nfeatures`
    strongest is among its tile's, so the image's strongest are then kept from
    far fewer descriptors, the same ones. They are sorted by response, then
    position and angle, so the order does not depend on the tiles either:
    RANSAC samples matches by index.
    """
    h, w = img.shape
    if max(h, w) <= SIFT_TILE_PX:
        return sift.detectAndCompute(img, None)

    keypoints, descriptors = [], []
    m = SIFT_TILE_MARGIN_PX
    for ty in range(0, h, SIFT_TILE_PX):
        for tx in range(0, w, SIFT_TILE_PX):
            # Tile core [tx, tx + tile) x [ty, ty + tile), read with the margin
            x0, y0 = max(0, tx - m), max(0, ty - m)
            x1, y1 = min(w, tx + SIFT_TILE_PX + m), min(h, ty + SIFT_TILE_PX + m)
            kps, desc = sift.detectAndCompute(img[y0:y1, x0:x1], None)
            if desc is None:
                continue
            for kp, d in zip(kps, desc):
                x, y = kp.pt[0] + x0, kp.pt[1] + y0
                if tx <= x < tx + SIFT_TILE_PX and ty <= y < ty + SIFT_TILE_PX:
                    keypoints.append(
                        cv2.KeyPoint(x, y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                    )
                    descriptors.append(d)
    if not keypoints:
        return [], None
    descriptors = np.array(descriptors, dtype=np.float32)
    order = np.lexsort(
        (
            [kp.angle for kp in keypoints],
            [kp.pt[0] for kp in keypoints],
            [kp.pt[1] for kp in keypoints],
            [-kp.response for kp in keypoints],
        )
    )
    if nfeatures:
        order = order[:nfeatures]
    return [keypoints[i] for i in order], descriptors[order]


def _sift_homography(
    mon: np.ndarray, ref: np.ndarray, sift_nfeatures: int, prior: Optional[np.ndarray]
) -> tuple[np.ndarray, int, int]:
    """SIFT + Lowe + cross-check + RANSAC homography mon → ref, between these two images.

    Returns (matrix, inliers, matches). Raises RuntimeError when SIFT or RANSAC fail.
    """
    mh, mw = mon.shape
    rh, rw = ref.shape
    logger.info(
        "SIFT feature matching: mon=%dx%d  ref=%dx%d  nfeatures=%s  contrast=%.3f  Lowe=%.2f  "
        "RANSAC=%.1fpx",
        mw,
        mh,
        rw,
        rh,
        sift_nfeatures or "unlimited",
        SIFT_CONTRAST_THRESHOLD,
        LOWE_RATIO,
        RANSAC_THRESHOLD_PX,
    )

    def detect(img: np.ndarray) -> tuple[list, Optional[np.ndarray]]:
        sift = cv2.SIFT_create(
            nfeatures=sift_nfeatures,
            contrastThreshold=SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=SIFT_EDGE_THRESHOLD,
        )
        return _detect_sift(sift, img, sift_nfeatures)

    kp_mon, desc_mon = detect(mon)
    kp_ref, desc_ref = detect(ref)

    if desc_mon is None or desc_ref is None:
        raise RuntimeError("SIFT found no descriptors in one or both images")
    if len(kp_mon) < MIN_MATCHES or len(kp_ref) < MIN_MATCHES:
        raise RuntimeError(
            f"Too few SIFT keypoints: mon={len(kp_mon)} ref={len(kp_ref)} " f"(need ≥{MIN_MATCHES})"
        )

    logger.info(
        "Keypoints detected: mon=%d  ref=%d, matched by %s",
        len(kp_mon),
        len(kp_ref),
        "brute force" if len(kp_mon) * len(kp_ref) <= BRUTE_FORCE_MAX_PAIRS else "FLANN KD-tree",
    )

    knn_fwd = _knn_match(desc_mon, desc_ref, k=2)

    # Lowe ratio test (mon → ref direction).
    lowe = []
    for pair in knn_fwd:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            lowe.append(m)

    # Mutual cross-check: for each kept mon→ref match, the ref keypoint's
    # nearest mon descriptor must point back to the same mon keypoint.
    if lowe:
        knn_bwd = _knn_match(desc_ref, desc_mon, k=1)
        bwd_best = {p[0].queryIdx: p[0].trainIdx for p in knn_bwd if p}
        good = [m for m in lowe if bwd_best.get(m.trainIdx) == m.queryIdx]
    else:
        good = []

    logger.info(
        "Matches: raw=%d  Lowe<%.2f=%d  mutual=%d",
        len(knn_fwd),
        LOWE_RATIO,
        len(lowe),
        len(good),
    )

    if len(good) < MIN_MATCHES:
        raise RuntimeError(
            f"Too few good matches after Lowe + cross-check: {len(good)} (need ≥{MIN_MATCHES})"
        )

    src_pts = np.array([kp_mon[m.queryIdx].pt for m in good], dtype=np.float32)
    dst_pts = np.array([kp_ref[m.trainIdx].pt for m in good], dtype=np.float32)

    if prior is not None:
        # Informational only: report how the matches sit relative to the prior
        predicted = _footprint_points(prior, src_pts)
        errors = np.linalg.norm(dst_pts - predicted, axis=1)
        logger.info(
            "Match error vs geotransform prior: median=%.1fpx  min=%.1fpx  max=%.1fpx",
            float(np.median(errors)),
            float(errors.min()),
            float(errors.max()),
        )

    matrix, inliers = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=10000,
        confidence=0.999,
    )
    if matrix is None:
        raise RuntimeError("RANSAC failed to estimate a homography")

    n_inliers = int(inliers.sum())
    logger.info(
        "RANSAC initial fit: %s  inliers=%d/%d (%.1f%%)",
        _decompose(matrix),
        n_inliers,
        len(good),
        100.0 * n_inliers / len(good),
    )
    return matrix, n_inliers, len(good)


def _footprint_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nx2 `points` mapped by the homography `matrix`."""
    mapped = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return mapped[:, :2] / mapped[:, 2:]


def detect_global_alignment(
    mon_arr: np.ndarray,
    ref_arr,
    prior: Optional[np.ndarray] = None,
    sift_nfeatures: int = SIFT_NFEATURES,
) -> GlobalAlignment:
    """Estimate a 2D homography (8 DOF) that maps mon pixels into ref pixels,
    via SIFT + RANSAC, then refined with ECC on Sobel gradient magnitudes.

    `ref_arr` is an array, or a RasterWindows reading only the windows used.

    `prior` (optional 3x3 homography from geotransforms) sets the working
    resolution and the search window, gives two more ECC starting points, the
    prior itself and its translation corrected by correlation, and bounds how
    far the result may depart from it. See the module docstring. The window
    extends GEOREF_SEARCH_FRACTION of the monitored footprint past it, and is
    doubled, up to the whole reference, while the result is doubtful: no
    plausible estimate, a gradient correlation under LOW_GRADIENT_CORRELATION,
    or a shift beyond EDGE_OF_WINDOW of the margin. The best result of the
    windows tried, by gradient correlation in the widest, is kept.

    `sift_nfeatures` is the number of SIFT keypoints kept in each image, the
    ones with the strongest response (OpenCV may keep a few more tied with the
    weakest one), 0 keeps them all. Limiting it bounds the time and memory of
    the brute-force matching on large images.
    """
    if sift_nfeatures < 0:
        raise ValueError(f"sift_nfeatures must be positive, or 0 for unlimited, got {sift_nfeatures}")

    if prior is None:
        return _align_without_prior(
            _preprocess(mon_arr), _preprocess(np.asarray(ref_arr)), sift_nfeatures
        )

    # Align in a window around the prior footprint, widened while the result
    # looks like the georeferencing error might exceed it
    attempts: list[_WindowAlignment] = []
    fraction = GEOREF_SEARCH_FRACTION
    while True:
        frame = _work_frame(mon_arr, ref_arr, prior, fraction)
        if frame is None:
            logger.warning("Prior footprint does not overlap ref: prior ignored")
            return _align_without_prior(
                _preprocess(mon_arr), _preprocess(np.asarray(ref_arr)), sift_nfeatures
            )
        logger.info(
            "Working images: mon=%dx%d  ref crop=%dx%d  search margin=%.0f px",
            frame.mon.shape[1],
            frame.mon.shape[0],
            frame.ref.shape[1],
            frame.ref.shape[0],
            frame.margin,
        )
        attempt = _align_in_window(frame, prior, sift_nfeatures)
        attempts.append(attempt)
        if not attempt.doubts:
            break
        if frame.covers_ref:
            logger.warning(
                "Alignment doubtful with the whole reference searched: %s",
                "; ".join(attempt.doubts),
            )
            break
        fraction *= 2
        logger.warning(
            "%s: widening the search margin to %.0f%% of the footprint",
            "; ".join(attempt.doubts),
            100 * fraction,
        )

    if len(attempts) == 1:
        return attempts[0].alignment
    # Each window's result, compared on the widest window's common pixels
    frame = attempts[-1].frame
    matrices = [frame.to_work(attempt.alignment.matrix) for attempt in attempts]
    scores, count = _gradient_correlation(frame, matrices)
    if np.isnan(scores).all():
        # The results do not even overlap: the narrower windows missed the scene
        logger.info("Search window results do not overlap; keeping the widest window's")
        return attempts[-1].alignment
    best = int(np.nanargmax(scores))
    logger.info(
        "Search windows compared on %d common px: %s, keeping window %d",
        count,
        "  ".join(f"{i + 1}={score:.4f}" for i, score in enumerate(scores)),
        best + 1,
    )
    return attempts[best].alignment


@dataclass
class _WindowAlignment:
    """Result of aligning within one search window, and why it may be wrong."""

    alignment: GlobalAlignment
    frame: _WorkFrame
    doubts: list = field(default_factory=list)  # empty when the result looks right


def _align_in_window(
    frame: _WorkFrame, prior: np.ndarray, sift_nfeatures: int
) -> _WindowAlignment:
    """Align mon within `frame`: SIFT, translation search and ECC from each start, then select.

    The result is doubted when no estimate is plausible, when it correlates
    poorly, or when it lands near the edge of the window: the true alignment
    might then lie beyond it.
    """
    # Starting points, as homographies between the working images
    starts: list[tuple[str, np.ndarray]] = []
    n_inliers = n_matches = 0
    try:
        ransac, n_inliers, n_matches = _sift_homography(
            frame.mon, frame.ref, sift_nfeatures, frame.to_work(prior)
        )
        starts.append(("RANSAC", ransac))
    except RuntimeError as e:
        logger.warning("SIFT estimate unavailable: %s", e)
    starts.append(("prior", frame.to_work(prior)))
    shifted = _search_translation(frame, frame.to_work(prior))
    if shifted is not None:
        starts.append(("shift", shifted))

    # A short ECC probe on Sobel gradients (sensor-invariant) from every start,
    # keeping the probes plausible as a georeferencing correction
    converged: list[tuple[str, np.ndarray, float]] = []
    origins: dict[str, np.ndarray] = dict(starts)
    for name, init in starts:
        refined, ecc_score = _refine_with_ecc(frame.mon, frame.ref, init, ECC_PROBE_ITERS)
        if refined is None:
            logger.warning("ECC probe from %s: failed", name)
            continue
        full = frame.to_full(refined)
        reason = _plausibility(full, prior, frame)
        logger.info(
            "ECC probe from %s: %s  ECC=%.4f%s",
            name,
            _decompose(full),
            ecc_score,
            f"  rejected: {reason}" if reason else "",
        )
        if reason is None:
            converged.append((name, full, ecc_score))

    if not converged:
        # Keep the best unrefined start: the translation search alone often
        # lands within a pixel where ECC does not converge
        plausible = [
            (name, frame.to_full(init))
            for name, init in starts
            if _plausibility(frame.to_full(init), prior, frame) is None
        ]
        scores, _ = _gradient_correlation(frame, [frame.to_work(m) for _, m in plausible])
        doubts = ["no ECC refinement converged to a plausible alignment"]
        if not plausible or np.isnan(scores).all():
            logger.warning("No plausible alignment; keeping the geotransform prior")
            alignment = GlobalAlignment(matrix=prior, n_inliers=n_inliers, n_matches=n_matches)
            return _WindowAlignment(alignment, frame, doubts)
        best = int(np.nanargmax(scores))
        name, matrix = plausible[best]
        logger.warning("No ECC refinement converged; keeping the %s estimate unrefined", name)
        alignment = GlobalAlignment(matrix=matrix, n_inliers=n_inliers, n_matches=n_matches)
        doubts += _window_doubts(matrix, prior, frame, scores[best])
        return _WindowAlignment(alignment, frame, doubts)

    scores, count = _gradient_correlation(frame, [frame.to_work(m) for _, m, _ in converged])
    if not np.isnan(scores).all():
        best = int(np.nanargmax(scores))
        logger.info(
            "Gradient correlation on %d common px: %s",
            count,
            "  ".join(f"{name}={score:.4f}" for (name, _, _), score in zip(converged, scores)),
        )
    else:
        # Too few pixels in common: fall back to each run's own ECC score
        best = int(np.argmax([ecc for _, _, ecc in converged]))
    name, matrix, ecc_score = converged[best]

    # Only the best probe's start is refined to convergence
    refined, final_score = _refine_with_ecc(frame.mon, frame.ref, origins[name])
    if refined is not None and _plausibility(frame.to_full(refined), prior, frame) is None:
        matrix, ecc_score = frame.to_full(refined), final_score
        converged[best] = (name, matrix, ecc_score)
    else:
        logger.warning("Full ECC from %s failed or is implausible; keeping its probe", name)
    logger.info("Selected alignment: ECC-refined from %s (ECC=%.4f)", name, ecc_score)
    final_scores, _ = _gradient_correlation(frame, [frame.to_work(matrix)])

    alignment = GlobalAlignment(
        matrix=matrix,
        n_inliers=n_inliers,
        n_matches=n_matches,
        candidates=converged,
    )
    return _WindowAlignment(alignment, frame, _window_doubts(matrix, prior, frame, final_scores[0]))


def _window_doubts(
    matrix: np.ndarray, prior: np.ndarray, frame: _WorkFrame, score: float
) -> list[str]:
    """Signs that the alignment found in `frame` may be cut short by its window."""
    doubts = []
    if np.isfinite(score) and score < LOW_GRADIENT_CORRELATION:
        doubts.append(f"gradient correlation {score:.2f} below {LOW_GRADIENT_CORRELATION}")
    position, _ = _at_mon_center(matrix, frame)
    prior_position, _ = _at_mon_center(prior, frame)
    shift = np.abs(position - prior_position)
    margin = frame.margin / frame.ref_to_work[0, 0]  # in ref full resolution px
    if shift.max() > EDGE_OF_WINDOW * margin:
        doubts.append(
            f"shift of {shift.max():.0f} ref px from the prior, beyond {EDGE_OF_WINDOW:.0%} "
            f"of the {margin:.0f} px search margin"
        )
    return doubts


def _align_without_prior(mon: np.ndarray, ref: np.ndarray, sift_nfeatures: int) -> GlobalAlignment:
    """SIFT + RANSAC on the whole images, refined with ECC, when no prior is available."""
    matrix, n_inliers, n_matches = _sift_homography(mon, ref, sift_nfeatures, None)

    converged: list[tuple[str, np.ndarray, float]] = []
    refined, ecc_score = _refine_with_ecc(mon, ref, matrix)
    if refined is None:
        logger.warning("ECC refinement failed; keeping RANSAC estimate")
    else:
        logger.info("ECC from RANSAC: %s  ECC=%.4f", _decompose(refined), ecc_score)
        converged.append(("RANSAC", refined, ecc_score))
        matrix = refined

    return GlobalAlignment(
        matrix=matrix,
        n_inliers=n_inliers,
        n_matches=n_matches,
        candidates=converged,
    )


def _decompose(matrix: np.ndarray) -> str:
    """Render a 3x3 homography as a compact diagnostic string.

    Reports the approximate similarity (rotation, mean scale, translation)
    extracted from the upper-left 2×2 and the translation column, plus the
    perspective row magnitude.
    """
    sx = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    sy = float(np.hypot(matrix[1, 0], matrix[1, 1]))
    rot = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    persp = float(np.hypot(matrix[2, 0], matrix[2, 1]))
    return (
        f"rot={rot:+.3f}°  sx={sx:.4f} sy={sy:.4f}  "
        f"tx={float(matrix[0,2]):+.2f} ty={float(matrix[1,2]):+.2f}  "
        f"persp={persp:.6f}"
    )


def _sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, normalized to [0, 1] float32.

    Gradient magnitude is far more sensor/band-invariant than raw intensity,
    which makes ECC work across modalities (different satellites/bands).
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = float(mag.max())
    return mag / m if m > 0 else mag


def _refine_with_ecc(
    mon_u8: np.ndarray,
    ref_u8: np.ndarray,
    init: np.ndarray,
    max_iters: int = ECC_MAX_ITERS,
) -> tuple[Optional[np.ndarray], float]:
    """Refine the mon → ref homography with cv2.findTransformECC.

    Strategy: pre-warp mon onto ref's canvas using `init`, then ask ECC to
    estimate the small residual 3x3 homography starting from identity,
    computing the correlation on Sobel gradient magnitudes. ECC's residual
    maps ref onto the pre-warped mon, so its inverse corrects `init`: composed
    the other way, as it once was, it doubled the starting error instead of
    removing it.

    Only ref around mon's pre-warped footprint, ECC_MARGIN_PX wider, is used:
    the correlation only counts mon's pixels, but ECC processes its whole
    template at every iteration, and a large ref crop around a small mon made
    it several times slower for the same result.

    `init` must be a 3x3 matrix.

    Returns (refined_3x3, ecc_score) on success, or (None, nan) on failure.
    """
    mh, mw = mon_u8.shape
    rh, rw = ref_u8.shape
    outer = np.array([[-0.5, -0.5], [mw - 0.5, -0.5], [mw - 0.5, mh - 0.5], [-0.5, mh - 0.5]])
    corners = _footprint_points(init, outer)
    if not np.isfinite(corners).all():
        return None, float("nan")
    x0 = int(np.clip(np.floor(corners[:, 0].min()) - ECC_MARGIN_PX, 0, rw))
    y0 = int(np.clip(np.floor(corners[:, 1].min()) - ECC_MARGIN_PX, 0, rh))
    x1 = int(np.clip(np.ceil(corners[:, 0].max()) + ECC_MARGIN_PX, 0, rw))
    y1 = int(np.clip(np.ceil(corners[:, 1].max()) + ECC_MARGIN_PX, 0, rh))
    if x1 <= x0 or y1 <= y0:
        return None, float("nan")
    crop = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    init_crop = crop @ init
    ref_u8 = ref_u8[y0:y1, x0:x1]

    warped_mon = cv2.warpPerspective(
        mon_u8,
        init_crop.astype(np.float32),
        (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    # Sobel and ECC's Gaussian read a pixel's neighbours: at the edge of the
    # data they see the jump to the zero fill, a gradient ref does not have
    valid = cv2.erode((warped_mon > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    if valid.sum() < MIN_VALID_PIXELS:
        logger.warning(
            "ECC skipped: pre-warped mon has only %d valid pixels (need >%d)",
            int(valid.sum()),
            MIN_VALID_PIXELS,
        )
        return None, float("nan")

    # Gradient-magnitude images — radiometry-invariant signal for ECC.
    template = _sobel_magnitude(ref_u8)
    image = _sobel_magnitude(warped_mon)
    residual = np.eye(3, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iters,
        ECC_EPS,
    )
    try:
        cc, residual = cv2.findTransformECC(
            template,
            image,
            residual,
            motionType=cv2.MOTION_HOMOGRAPHY,
            criteria=criteria,
            inputMask=valid * 255,
            gaussFiltSize=5,
        )
    except cv2.error as e:
        logger.warning("findTransformECC raised: %s", e)
        return None, float("nan")

    # ECC's warp maps template (ref) coordinates to input (pre-warped mon) ones,
    # so the mon → ref correction is its inverse
    final = np.linalg.inv(crop) @ np.linalg.inv(residual.astype(np.float64)) @ init_crop
    return final, float(cc)


def _gdal_dtype_for(np_dtype: np.dtype) -> int:
    return _NUMPY_TO_GDAL_DTYPE.get(np.dtype(np_dtype), gdal.GDT_Float32)


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    x_min: float,
    y_max: float,
    x_res: float,
    y_res: float,
    projection: str,
    nodata: Optional[float],
    gdal_dtype: Optional[int] = None,
) -> None:
    driver = gdal.GetDriverByName("GTiff")
    h, w = data.shape
    dtype = gdal_dtype if gdal_dtype is not None else _gdal_dtype_for(data.dtype)
    dataset = driver.Create(str(path), w, h, 1, dtype, options=["COMPRESS=LZW"])
    if projection:
        dataset.SetProjection(projection)
    dataset.SetGeoTransform((x_min, x_res, 0, y_max, 0, y_res))
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    dataset.FlushCache()
    band = None
    dataset = None


@dataclass
class OutputFrame:
    """North-up grid the aligned outputs are written on, and the warp of mon onto it."""

    projection: str
    x_min: float
    y_max: float
    x_res: float
    y_res: float
    width: int
    height: int
    from_mon: np.ndarray  # 3x3 map from mon pixels to output pixels


def _geo_matrix(geo_transform: tuple) -> np.ndarray:
    """3x3 map of a GDAL geotransform, from pixel corner coordinates to the CRS."""
    g = geo_transform
    return np.array([[g[1], g[2], g[0]], [g[4], g[5], g[3]], [0.0, 0.0, 1.0]])


def _frame_in_mon_crs(
    matrix: np.ndarray, georefs: np.ndarray, mon_valid: np.ndarray, monitored: GdalRasterImage
) -> OutputFrame:
    """Grid over mon's corrected valid footprint on mon's own grid, in mon's CRS.

    `matrix` maps mon pixels to ref pixels where they truly are, `georefs`
    where the georeferencing puts them. A point of mon's CRS lands in ref
    through `georefs`, then on the mon pixel imaging it through the inverse
    of `matrix`.

    A north-up mon keeps its grid: same pixel size and edges, shifted by whole
    pixels to cover the corrected footprint, so a correction of whole pixels
    moves the georeferencing and leaves the pixels untouched. A rotated or
    mirrored mon gets a north-up grid from its origin, at its pixel area.
    """
    geo = monitored.geo_transform
    if geo[1] > 0 and geo[2] == 0 and geo[4] == 0 and geo[5] < 0:
        x_res, y_res = geo[1], geo[5]
    else:
        side = float(np.sqrt(abs(geo[1] * geo[5] - geo[2] * geo[4])))
        x_res, y_res = side, -side
    # mon pixels → pixels of the north-up grid from mon's origin, in OpenCV's
    # center coordinates: geotransforms map corners, half a pixel out
    grid = (geo[0], x_res, 0.0, geo[3], 0.0, y_res)
    to_grid = np.linalg.inv(_geo_matrix(grid)) @ _geo_matrix(geo)
    half = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    corrected = np.linalg.inv(half) @ to_grid @ half @ np.linalg.inv(georefs) @ matrix

    edges = _footprint_points(corrected, _valid_outline(mon_valid) - 0.5) + 0.5
    if not np.isfinite(edges).all():
        raise RuntimeError("The aligned monitored image footprint is not finite")
    x0, y0 = np.floor(edges.min(axis=0)).astype(int)
    x1, y1 = np.ceil(edges.max(axis=0)).astype(int)
    offset = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    return OutputFrame(
        monitored.projection,
        geo[0] + x0 * x_res,
        geo[3] + y0 * y_res,
        x_res,
        y_res,
        int(x1 - x0),
        int(y1 - y0),
        offset @ corrected,
    )


@dataclass
class OutputGrid:
    """Pixel grid nested in ref's grid, for the aligned outputs without georeferencing prior.

    `factor` output pixels span one ref pixel on each axis; the grid covers
    ref pixels [x0, x0 + width / factor) x [y0, y0 + height / factor).
    """

    factor: int
    x0: int
    y0: int
    width: int
    height: int

    @property
    def from_ref(self) -> np.ndarray:
        """3x3 map from ref pixels to output pixels."""
        offset = np.array([[1.0, 0.0, -self.x0], [0.0, 1.0, -self.y0], [0.0, 0.0, 1.0]])
        return _pixel_scale(self.factor, self.factor) @ offset


def _valid_outline(valid: np.ndarray, step: int = 16) -> np.ndarray:
    """Convex outline of the valid pixels, as the pixel edges enclosing them (Nx2, x y).

    Rows are sampled every `step` pixels and each sample is widened by a step,
    so the outline may exceed the data by up to `step` pixels but never cuts
    into it. Falls back to the image edges when no pixel is valid.
    """
    h, w = valid.shape
    points = []
    for y in range(0, h, step):
        band = valid[y : y + step].any(axis=0)
        cols = np.flatnonzero(band)
        if cols.size:
            bottom = min(y + step, h)
            points += [(cols[0], y), (cols[-1] + 1, y), (cols[0], bottom), (cols[-1] + 1, bottom)]
    if not points:
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    hull = cv2.convexHull(np.array(points, dtype=np.float32))
    return hull.reshape(-1, 2).astype(float)


def _output_grid(
    matrix: np.ndarray, mon_valid: np.ndarray, ref_shape: tuple
) -> OutputGrid:
    """Grid over mon's valid footprint in ref, `factor` times finer than ref.

    `factor` is the smallest integer that does not coarsen mon beyond
    OUTPUT_RESOLUTION_TOLERANCE, from mon's pixel size in ref pixels under
    `matrix` at its center: a monitored image 7.2 times finer than ref gets a
    grid 8 times finer, one 4 times finer a grid 4 times finer even when the
    estimate says 4.01. A coarser mon gets ref's own grid. Nesting in ref's
    grid keeps both outputs on ref's pixel edges.
    """
    mh, mw = mon_valid.shape
    rh, rw = ref_shape
    center = np.array([mw / 2, mh / 2, 1.0])
    p = matrix @ center
    jacobian = (matrix[:2, :2] * p[2] - np.outer(p[:2], matrix[2, :2])) / p[2] ** 2
    mon_px = float(np.sqrt(abs(np.linalg.det(jacobian))))  # mon pixel side in ref pixels
    factor = max(1, int(np.ceil((1 - OUTPUT_RESOLUTION_TOLERANCE) / mon_px)))

    # The valid data's outline in ref, from pixel edges to OpenCV's centers on
    # integers and back: edges sit half a pixel out of centers
    edges = _footprint_points(matrix, _valid_outline(mon_valid) - 0.5) + 0.5
    x0 = int(np.clip(np.floor(edges[:, 0].min()), 0, rw))
    y0 = int(np.clip(np.floor(edges[:, 1].min()), 0, rh))
    x1 = int(np.clip(np.ceil(edges[:, 0].max()), 0, rw))
    y1 = int(np.clip(np.ceil(edges[:, 1].max()), 0, rh))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("The aligned monitored image does not overlap the reference")
    return OutputGrid(factor, x0, y0, (x1 - x0) * factor, (y1 - y0) * factor)


def _frame_in_ref(
    matrix: np.ndarray, mon_valid: np.ndarray, reference: GdalRasterImage
) -> OutputFrame:
    """The grid of _output_grid(), georeferenced in ref's CRS."""
    grid = _output_grid(matrix, mon_valid, (reference.y_size, reference.x_size))
    return OutputFrame(
        reference.projection,
        reference.x_min + grid.x0 * reference.x_res,
        reference.y_max + grid.y0 * reference.y_res,
        reference.x_res / grid.factor,
        reference.y_res / grid.factor,
        grid.width,
        grid.height,
        grid.from_ref @ matrix,
    )


def _to_dtype(values: np.ndarray, dtype) -> np.ndarray:
    """Interpolated `values` back to the image's `dtype`, rounded and clipped for integers.

    A plain astype truncates: 0.9 became 0, a -0.5 DN bias on every pixel.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


def apply_global_alignment(
    monitored: GdalRasterImage,
    reference: GdalRasterImage,
    mask: Optional[GdalRasterImage],
    out_dir: Path,
    sift_nfeatures: int = SIFT_NFEATURES,
) -> tuple[GdalRasterImage, Optional[GdalRasterImage], GlobalAlignment]:
    """Detect the homography, apply to monitored (and mask), render on mon's grid in mon's CRS.

    The outputs keep mon's CRS and grid, over its corrected valid footprint,
    see _frame_in_mon_crs(). Without a georeferencing prior, an image without
    CRS or a failed reprojection, they are rendered on a grid nested in ref's
    instead, covering mon's valid footprint at a whole fraction of ref's pixel
    size that keeps mon's resolution, see _output_grid(), and georeferenced in
    ref's CRS.

    Returns (aligned_mon, aligned_mask, alignment_info), the rasters written
    to `out_dir`. `sift_nfeatures` limits the SIFT keypoints kept per image,
    see detect_global_alignment().
    """
    mon_arr = monitored.array
    # Read window by window: with a prior only the search window is ever read
    ref_arr = RasterWindows(reference)

    prior = _prior_from_georefs(monitored, reference)
    if prior is not None:
        logger.info(
            "Geotransform prior available: %s",
            _decompose(prior),
        )
    else:
        logger.info("No geotransform prior (unreferenced image, or reprojection failed)")

    alignment = detect_global_alignment(
        mon_arr, ref_arr, prior=prior, sift_nfeatures=sift_nfeatures
    )

    # The homography maps mon pixel coords → ref pixel coords. The outputs are
    # rendered on mon's own grid in its CRS, or, without georeferencing to
    # bring ref pixels back to mon's CRS, on a grid nested in ref's.
    mon_valid = _valid_pixels(mon_arr)
    if monitored.no_data_value is not None:
        mon_valid &= mon_arr != monitored.no_data_value
    if prior is not None:
        frame = _frame_in_mon_crs(alignment.matrix, prior, mon_valid, monitored)
    else:
        logger.warning("Output georeferenced in ref's CRS: no georeferencing to keep mon's")
        frame = _frame_in_ref(alignment.matrix, mon_valid, reference)
    out_size = (frame.width, frame.height)
    warp_m = frame.from_mon
    logger.info(
        "Output grid: %dx%d px of %.6g x %.6g from (%.6f, %.6f)",
        frame.width,
        frame.height,
        frame.x_res,
        abs(frame.y_res),
        frame.x_min,
        frame.y_max,
    )

    border_mon = float(monitored.no_data_value) if monitored.no_data_value is not None else 0.0
    aligned_mon = _to_dtype(
        cv2.warpPerspective(
            mon_arr.astype(np.float32),
            warp_m,
            out_size,
            flags=cv2.INTER_LINEAR,
            borderValue=border_mon,
        ),
        mon_arr.dtype,
    )

    mon_stem = Path(monitored.file_name).stem
    mon_suffix = Path(monitored.file_name).suffix or ".tif"
    mon_out = out_dir / f"{mon_stem}_global_aligned{mon_suffix}"

    _write_geotiff(
        mon_out,
        aligned_mon,
        frame.x_min,
        frame.y_max,
        frame.x_res,
        frame.y_res,
        frame.projection,
        monitored.no_data_value,
    )
    aligned_mask = None
    if mask is not None:
        warped_mask = cv2.warpPerspective(
            mask.array.astype(np.uint8),
            warp_m,
            out_size,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        mask_stem = Path(mask.file_name).stem
        mask_suffix = Path(mask.file_name).suffix or ".tif"
        mask_out = out_dir / f"{mask_stem}_global_aligned{mask_suffix}"
        _write_geotiff(
            mask_out,
            warped_mask,
            frame.x_min,
            frame.y_max,
            frame.x_res,
            frame.y_res,
            frame.projection,
            None,
            gdal_dtype=gdal.GDT_Byte,
        )
        aligned_mask = GdalRasterImage(str(mask_out))

    return GdalRasterImage(str(mon_out)), aligned_mask, alignment
