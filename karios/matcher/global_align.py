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
    5. Match with BFMatcher(NORM_L2), apply Lowe's ratio test + mutual
       (cross-check) filtering for robustness.
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
       cv2.warpPerspective, rendered over mon's footprint in ref at a whole
       fraction of ref's pixel size, fine enough to keep mon's resolution;
       ref is resampled onto the same grid.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from osgeo import gdal

from karios.core.image import GdalRasterImage
from karios.core.radiometry import to_uint8

logger = logging.getLogger(__name__)

SIFT_NFEATURES = 0  # default max number of keypoints kept per image, 0 = unlimited
SIFT_CONTRAST_THRESHOLD = 0.02  # default 0.04; lower → more keypoints in low-contrast regions
SIFT_EDGE_THRESHOLD = 10
LOWE_RATIO = 0.75
RANSAC_THRESHOLD_PX = 3.0
MIN_MATCHES = 4  # cv2.findHomography needs ≥4 point pairs; more = robuster
ECC_MAX_ITERS = 200
ECC_EPS = 1e-6
MIN_VALID_PIXELS = 1000  # fewest valid pixels an ECC run or a correlation score needs

# Georeferencing correction limits, with a prior. The search window extends
# past mon's prior footprint by this share of its size on each side.
GEOREF_SEARCH_FRACTION = 0.5
MAX_SCALE_CHANGE = 1.5  # largest scale factor from the prior, either way
MAX_ANISOTROPY = 1.3  # largest ratio between the scale factors of both axes
MAX_ROTATION_DEG = 30.0
MIN_SHIFT_BLOCK_PX = 32  # smallest central block the translation search correlates
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
    # converged — including the one chosen as `matrix`. Lets callers write
    # alternative warped outputs for visual A/B comparison.
    candidates: list = field(default_factory=list)

    @property
    def score(self) -> float:
        """RANSAC inlier ratio in [0, 1]."""
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


def _preprocess(arr: np.ndarray) -> np.ndarray:
    """uint8 stretch + CLAHE to equalize radiometry across the two images."""
    img = to_uint8(arr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _prior_from_georefs(
    monitored: GdalRasterImage, reference: GdalRasterImage
) -> Optional[np.ndarray]:
    """Build the 3x3 homography mon_pixel → ref_pixel implied by the two
    geotransforms, assuming both images are georeferenced in the same CRS
    and north-up (zero skew terms in their geotransforms — true for almost
    all satellite GeoTIFFs).

    Returns None when no usable prior can be built (missing projection or
    mismatched CRS).
    """
    if not monitored.projection or not reference.projection:
        return None
    try:
        if not monitored.spatial_ref.IsSame(reference.spatial_ref):
            return None
    except Exception:
        return None
    sx = monitored.x_res / reference.x_res
    sy = monitored.y_res / reference.y_res
    tx = (monitored.x_min - reference.x_min) / reference.x_res
    ty = (monitored.y_max - reference.y_max) / reference.y_res
    # Geotransforms place pixel corners, OpenCV pixel centers: see _pixel_scale()
    return np.array(
        [[sx, 0.0, tx + (sx - 1) / 2], [0.0, sy, ty + (sy - 1) / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass
class _WorkFrame:
    """mon and a crop of ref at a common resolution, with the maps back to full resolution."""

    mon: np.ndarray  # uint8 mon at the working resolution
    mon_valid: np.ndarray  # bool, pixels of `mon` fully covered by valid data
    ref: np.ndarray  # uint8 crop of ref at the working resolution
    mon_to_work: np.ndarray  # 3x3, mon full resolution px -> `mon` px
    ref_to_work: np.ndarray  # 3x3, ref full resolution px -> `ref` crop px
    margin: float  # search margin around mon's prior footprint, in `ref` px

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
    mon_u8: np.ndarray, mon_valid: np.ndarray, ref_u8: np.ndarray, prior: np.ndarray
) -> Optional[_WorkFrame]:
    """Bring mon and ref to the coarser resolution and crop ref around mon's prior footprint.

    Returns None when the prior footprint does not overlap ref.
    """
    # Pixel size of mon relative to ref, from the prior's area scale
    scale = float(np.sqrt(abs(np.linalg.det(prior[:2, :2]))))
    mh, mw = mon_u8.shape
    rh, rw = ref_u8.shape

    mon_work, valid_work = mon_u8, mon_valid
    if scale < 1:
        size = (max(1, round(mw * scale)), max(1, round(mh * scale)))
        mon_work = cv2.resize(mon_u8, size, interpolation=cv2.INTER_AREA)
        # A working pixel is valid only when every mon pixel under it is
        coverage = cv2.resize(mon_valid.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        valid_work = coverage > 0.999
    mon_to_work = _pixel_scale(mon_work.shape[1] / mw, mon_work.shape[0] / mh)

    ref_work = ref_u8
    if scale > 1:
        size = (max(1, round(rw / scale)), max(1, round(rh / scale)))
        ref_work = cv2.resize(ref_u8, size, interpolation=cv2.INTER_AREA)
    ref_scale = _pixel_scale(ref_work.shape[1] / rw, ref_work.shape[0] / rh)

    corners = _footprint(ref_scale @ prior, mw, mh)
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    margin = GEOREF_SEARCH_FRACTION * max(x1 - x0, y1 - y0)
    cx0, cy0 = max(0, int(np.floor(x0 - margin))), max(0, int(np.floor(y0 - margin)))
    cx1 = min(ref_work.shape[1], int(np.ceil(x1 + margin)))
    cy1 = min(ref_work.shape[0], int(np.ceil(y1 + margin)))
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    crop = np.array([[1.0, 0.0, -cx0], [0.0, 1.0, -cy0], [0.0, 0.0, 1.0]])
    return _WorkFrame(
        mon=mon_work,
        mon_valid=valid_work,
        ref=ref_work[cy0:cy1, cx0:cx1],
        mon_to_work=mon_to_work,
        ref_to_work=crop @ ref_scale,
        margin=margin,
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


def _plausibility(matrix: np.ndarray, prior: np.ndarray, frame: _WorkFrame) -> Optional[str]:
    """Why `matrix` is too far from `prior` to be a georeferencing correction, None if it is not.

    Compares the linear parts at mon's center, where the homography is
    linearized, and the positions of mon's center.
    """
    mh, mw = frame.mon.shape
    center = np.linalg.inv(frame.mon_to_work) @ np.array([mw / 2, mh / 2, 1.0])

    def local(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position and 2x2 Jacobian of h at mon's center."""
        p = h @ center
        w = p[2]
        jacobian = (h[:2, :2] * w - np.outer(p[:2], h[2, :2])) / w**2
        return p[:2] / w, jacobian

    pos, jacobian = local(matrix)
    prior_pos, prior_jacobian = local(prior)
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

    sift = cv2.SIFT_create(
        nfeatures=sift_nfeatures,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=SIFT_EDGE_THRESHOLD,
    )
    kp_mon, desc_mon = sift.detectAndCompute(mon, None)
    kp_ref, desc_ref = sift.detectAndCompute(ref, None)

    if desc_mon is None or desc_ref is None:
        raise RuntimeError("SIFT found no descriptors in one or both images")
    if len(kp_mon) < MIN_MATCHES or len(kp_ref) < MIN_MATCHES:
        raise RuntimeError(
            f"Too few SIFT keypoints: mon={len(kp_mon)} ref={len(kp_ref)} " f"(need ≥{MIN_MATCHES})"
        )

    logger.info("Keypoints detected: mon=%d  ref=%d", len(kp_mon), len(kp_ref))

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_fwd = matcher.knnMatch(desc_mon, desc_ref, k=2)

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
        knn_bwd = matcher.knnMatch(desc_ref, desc_mon, k=1)
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
    ref_arr: np.ndarray,
    prior: Optional[np.ndarray] = None,
    sift_nfeatures: int = SIFT_NFEATURES,
) -> GlobalAlignment:
    """Estimate a 2D homography (8 DOF) that maps mon pixels into ref pixels,
    via SIFT + RANSAC, then refined with ECC on Sobel gradient magnitudes.

    `prior` (optional 3x3 homography from geotransforms) sets the working
    resolution and the search window, gives two more ECC starting points, the
    prior itself and its translation corrected by correlation, and bounds how
    far the result may depart from it. See the module docstring.

    `sift_nfeatures` is the number of SIFT keypoints kept in each image, the
    ones with the strongest response (OpenCV may keep a few more tied with the
    weakest one), 0 keeps them all. Limiting it bounds the time and memory of
    the brute-force matching on large images.
    """
    if sift_nfeatures < 0:
        raise ValueError(f"sift_nfeatures must be positive, or 0 for unlimited, got {sift_nfeatures}")

    mon = _preprocess(mon_arr)
    ref = _preprocess(ref_arr)

    frame = None
    if prior is not None:
        frame = _work_frame(mon, _valid_pixels(mon_arr), ref, prior)
        if frame is None:
            logger.warning("Prior footprint does not overlap ref: prior ignored")
            prior = None
        else:
            logger.info(
                "Working images: mon=%dx%d  ref crop=%dx%d  search margin=%.0f px",
                frame.mon.shape[1],
                frame.mon.shape[0],
                frame.ref.shape[1],
                frame.ref.shape[0],
                frame.margin,
            )

    if frame is None:
        return _align_without_prior(mon, ref, sift_nfeatures)

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

    # ECC refinement on Sobel gradients (sensor-invariant) from every start,
    # keeping the refined estimates plausible as a georeferencing correction
    converged: list[tuple[str, np.ndarray, float]] = []
    for name, init in starts:
        refined, ecc_score = _refine_with_ecc(frame.mon, frame.ref, init)
        if refined is None:
            logger.warning("ECC from %s: failed", name)
            continue
        full = frame.to_full(refined)
        reason = _plausibility(full, prior, frame)
        logger.info(
            "ECC from %s: %s  ECC=%.4f%s",
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
        if not plausible or np.isnan(scores).all():
            logger.warning("No plausible alignment; keeping the geotransform prior")
            return GlobalAlignment(matrix=prior, n_inliers=n_inliers, n_matches=n_matches)
        name, matrix = plausible[int(np.nanargmax(scores))]
        logger.warning("No ECC refinement converged; keeping the %s estimate unrefined", name)
        return GlobalAlignment(matrix=matrix, n_inliers=n_inliers, n_matches=n_matches)

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
    logger.info("Selected alignment: ECC-refined from %s (ECC=%.4f)", name, ecc_score)

    return GlobalAlignment(
        matrix=matrix,
        n_inliers=n_inliers,
        n_matches=n_matches,
        candidates=converged,
    )


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
) -> tuple[Optional[np.ndarray], float]:
    """Refine the mon → ref homography with cv2.findTransformECC.

    Strategy: pre-warp mon onto ref's canvas using `init`, then ask ECC to
    estimate the small residual 3x3 homography starting from identity,
    computing the correlation on Sobel gradient magnitudes.

    `init` must be a 3x3 matrix.

    Returns (refined_3x3, ecc_score) on success, or (None, nan) on failure.
    """
    rh, rw = ref_u8.shape
    warped_mon = cv2.warpPerspective(
        mon_u8,
        init.astype(np.float32),
        (rw, rh),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    valid = (warped_mon > 0).astype(np.uint8)
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
        ECC_MAX_ITERS,
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

    final = residual.astype(np.float64) @ init.astype(np.float64)
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
class OutputGrid:
    """Pixel grid the aligned outputs are written on, nested in ref's grid.

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


def _resample_ref(ref_arr: np.ndarray, grid: OutputGrid) -> np.ndarray:
    """ref on `grid`: cropped when it has ref's pixel size, cubic interpolation otherwise."""
    rows = slice(grid.y0, grid.y0 + grid.height // grid.factor)
    cols = slice(grid.x0, grid.x0 + grid.width // grid.factor)
    if grid.factor == 1:
        return ref_arr[rows, cols].copy()
    upsampled = cv2.warpPerspective(
        ref_arr.astype(np.float32),
        grid.from_ref.astype(np.float32),
        (grid.width, grid.height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if np.issubdtype(ref_arr.dtype, np.integer):
        # Cubic interpolation overshoots at edges: keep inside the integer type
        info = np.iinfo(ref_arr.dtype)
        upsampled = np.clip(np.round(upsampled), info.min, info.max)
    return upsampled.astype(ref_arr.dtype)


def apply_global_alignment(
    monitored: GdalRasterImage,
    reference: GdalRasterImage,
    mask: Optional[GdalRasterImage],
    out_dir: Path,
    sift_nfeatures: int = SIFT_NFEATURES,
) -> tuple[
    GdalRasterImage,
    GdalRasterImage,
    Optional[GdalRasterImage],
    GlobalAlignment,
]:
    """Detect the homography, apply to monitored (and mask), render over mon's footprint in ref.

    The outputs share a grid nested in ref's, covering mon's footprint at a
    whole fraction of ref's pixel size that keeps mon's resolution, see
    _output_grid(). ref is resampled onto it.

    Returns (aligned_mon, ref_on_grid, aligned_mask, alignment_info). The
    new rasters are written to `out_dir` so the rest of the pipeline can
    operate on them as if they were the originals. `sift_nfeatures` limits the
    SIFT keypoints kept per image, see detect_global_alignment().
    """
    mon_arr = monitored.array
    ref_arr = reference.array

    prior = _prior_from_georefs(monitored, reference)
    if prior is not None:
        logger.info(
            "Geotransform prior available: %s",
            _decompose(prior),
        )
    else:
        logger.info("No geotransform prior (CRS mismatch or unreferenced)")

    alignment = detect_global_alignment(
        mon_arr, ref_arr, prior=prior, sift_nfeatures=sift_nfeatures
    )

    # The homography maps mon pixel coords → ref pixel coords. Both outputs
    # are rendered on a grid over mon's footprint nested in ref's, fine enough
    # to keep mon's resolution, so they overlay directly without losing detail.
    mon_valid = _valid_pixels(mon_arr)
    if monitored.no_data_value is not None:
        mon_valid &= mon_arr != monitored.no_data_value
    grid = _output_grid(alignment.matrix, mon_valid, ref_arr.shape)
    out_size = (grid.width, grid.height)
    warp_m = grid.from_ref @ alignment.matrix
    x_min = reference.x_min + grid.x0 * reference.x_res
    y_max = reference.y_max + grid.y0 * reference.y_res
    x_res = reference.x_res / grid.factor
    y_res = reference.y_res / grid.factor
    logger.info(
        "Output grid: %dx%d px of %.3f x %.3f, %d per ref pixel, over ref pixels x=%d-%d y=%d-%d",
        grid.width,
        grid.height,
        x_res,
        abs(y_res),
        grid.factor,
        grid.x0,
        grid.x0 + grid.width // grid.factor,
        grid.y0,
        grid.y0 + grid.height // grid.factor,
    )

    border_mon = float(monitored.no_data_value) if monitored.no_data_value is not None else 0.0
    aligned_mon = cv2.warpPerspective(
        mon_arr.astype(np.float32),
        warp_m,
        out_size,
        flags=cv2.INTER_LINEAR,
        borderValue=border_mon,
    ).astype(mon_arr.dtype)

    mon_stem = Path(monitored.file_name).stem
    mon_suffix = Path(monitored.file_name).suffix or ".tif"
    ref_stem = Path(reference.file_name).stem
    ref_suffix = Path(reference.file_name).suffix or ".tif"
    mon_out = out_dir / f"{mon_stem}_global_aligned{mon_suffix}"
    ref_out = out_dir / f"{ref_stem}_global_aligned{ref_suffix}"

    _write_geotiff(
        mon_out,
        aligned_mon,
        x_min,
        y_max,
        x_res,
        y_res,
        reference.projection,
        monitored.no_data_value,
    )
    _write_geotiff(
        ref_out,
        _resample_ref(ref_arr, grid),
        x_min,
        y_max,
        x_res,
        y_res,
        reference.projection,
        reference.no_data_value,
    )

    # Write every refinement candidate as a sibling file so the user can A/B
    # them in QGIS. ECC scores are unreliable on weakly-correlated cross-sensor
    # imagery, so the algorithm's "best" pick may not be the visually best one.
    for cand_name, cand_matrix, cand_ecc in alignment.candidates:
        if cand_matrix is alignment.matrix or np.allclose(cand_matrix, alignment.matrix):
            continue
        cand_warped = cv2.warpPerspective(
            mon_arr.astype(np.float32),
            (grid.from_ref @ cand_matrix).astype(np.float32),
            out_size,
            flags=cv2.INTER_LINEAR,
            borderValue=border_mon,
        ).astype(mon_arr.dtype)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", cand_name).strip("_")
        cand_path = out_dir / f"{mon_stem}_global_aligned__{safe}_ecc{cand_ecc:.3f}{mon_suffix}"
        _write_geotiff(
            cand_path,
            cand_warped,
            x_min,
            y_max,
            x_res,
            y_res,
            reference.projection,
            monitored.no_data_value,
        )
        logger.info("Wrote alternative: %s", cand_path.name)

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
            x_min,
            y_max,
            x_res,
            y_res,
            reference.projection,
            None,
            gdal_dtype=gdal.GDT_Byte,
        )
        aligned_mask = GdalRasterImage(str(mask_out))

    return (
        GdalRasterImage(str(mon_out)),
        GdalRasterImage(str(ref_out)),
        aligned_mask,
        alignment,
    )
