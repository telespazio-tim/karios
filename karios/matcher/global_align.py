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
"""Global homography preprocessing via SIFT feature matching.

Pipeline:
    1. Preprocess both mon and ref to uint8 with CLAHE (equalizes radiometry
       between sensors).
    2. Detect SIFT keypoints + 128-dim float descriptors on both.
    3. Match with BFMatcher(NORM_L2), apply Lowe's ratio test + mutual
       (cross-check) filtering for robustness.
    4. Fit a 2D homography (8 DOF) using cv2.findHomography + RANSAC.
    5. Refine with cv2.findTransformECC(MOTION_HOMOGRAPHY) on Sobel gradient
       magnitudes (sensor-invariant), trying several initial estimates and
       keeping the one with the highest ECC score.
    6. Apply the resulting 3x3 homography to mon (and mask) via
       cv2.warpPerspective, rendered onto ref's canvas.
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

logger = logging.getLogger(__name__)

SIFT_NFEATURES = 0  # 0 = unlimited
SIFT_CONTRAST_THRESHOLD = 0.02  # default 0.04; lower → more keypoints in low-contrast regions
SIFT_EDGE_THRESHOLD = 10
LOWE_RATIO = 0.75
RANSAC_THRESHOLD_PX = 3.0
MIN_MATCHES = 4  # cv2.findHomography needs ≥4 point pairs; more = robuster
ECC_MAX_ITERS = 200
ECC_EPS = 1e-6


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


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    # Percentile stretch is more robust than min/max to a few outliers.
    lo, hi = np.percentile(a[finite], (2.0, 98.0))
    if hi > lo:
        a = np.clip(((a - lo) / (hi - lo)) * 255.0, 0, 255)
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)


def _preprocess(arr: np.ndarray) -> np.ndarray:
    """uint8 stretch + CLAHE to equalize radiometry across the two images."""
    img = _to_uint8(arr)
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
    return np.array(
        [[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def detect_global_alignment(
    mon_arr: np.ndarray,
    ref_arr: np.ndarray,
    prior: Optional[np.ndarray] = None,
) -> GlobalAlignment:
    """Estimate a 2D homography (8 DOF) that maps mon pixels into ref pixels,
    via SIFT + RANSAC, then refined with ECC on Sobel gradient magnitudes.

    `prior` (optional 3x3 homography from geotransforms) provides an extra
    ECC starting point but is not used to filter matches.
    """
    mon = _preprocess(mon_arr)
    ref = _preprocess(ref_arr)

    mh, mw = mon.shape
    rh, rw = ref.shape
    logger.info(
        "SIFT feature matching: mon=%dx%d  ref=%dx%d  contrast=%.3f  Lowe=%.2f  RANSAC=%.1fpx",
        mw, mh, rw, rh,
        SIFT_CONTRAST_THRESHOLD, LOWE_RATIO, RANSAC_THRESHOLD_PX,
    )

    sift = cv2.SIFT_create(
        nfeatures=SIFT_NFEATURES,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=SIFT_EDGE_THRESHOLD,
    )
    kp_mon, desc_mon = sift.detectAndCompute(mon, None)
    kp_ref, desc_ref = sift.detectAndCompute(ref, None)

    if desc_mon is None or desc_ref is None:
        raise RuntimeError("SIFT found no descriptors in one or both images")
    if len(kp_mon) < MIN_MATCHES or len(kp_ref) < MIN_MATCHES:
        raise RuntimeError(
            f"Too few SIFT keypoints: mon={len(kp_mon)} ref={len(kp_ref)} "
            f"(need ≥{MIN_MATCHES})"
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
        len(knn_fwd), LOWE_RATIO, len(lowe), len(good),
    )

    if len(good) < MIN_MATCHES:
        raise RuntimeError(
            f"Too few good matches after Lowe + cross-check: {len(good)} (need ≥{MIN_MATCHES})"
        )

    src_pts = np.array([kp_mon[m.queryIdx].pt for m in good], dtype=np.float32)
    dst_pts = np.array([kp_ref[m.trainIdx].pt for m in good], dtype=np.float32)

    if prior is not None:
        # Informational only: report how the matches sit relative to the prior
        # (the prior's upper-left 2x3 captures translation + scale; perspective
        # rows are zero in geotransform priors).
        predicted = (prior[:2, :2] @ src_pts.T).T + prior[:2, 2]
        errors = np.linalg.norm(dst_pts - predicted, axis=1)
        logger.info(
            "Match error vs geotransform prior: median=%.1fpx  min=%.1fpx  max=%.1fpx",
            float(np.median(errors)), float(errors.min()), float(errors.max()),
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
        _decompose(matrix), n_inliers, len(good), 100.0 * n_inliers / len(good),
    )

    # ECC refinement on Sobel gradients (sensor-invariant), tried from every
    # available starting point. The highest ECC wins.
    candidates: list[tuple[str, np.ndarray]] = [("RANSAC", matrix)]
    if prior is not None:
        candidates.append(("prior", prior))

    converged: list[tuple[str, np.ndarray, float]] = []
    best_matrix: Optional[np.ndarray] = None
    best_ecc = -np.inf
    best_source = ""
    for name, init in candidates:
        refined, ecc_score = _refine_with_ecc(mon, ref, init)
        if refined is None:
            logger.warning("ECC from %s: failed", name)
            continue
        logger.info(
            "ECC from %s: %s  ECC=%.4f",
            name, _decompose(refined), ecc_score,
        )
        converged.append((name, refined, ecc_score))
        if ecc_score > best_ecc:
            best_ecc = ecc_score
            best_matrix = refined
            best_source = name

    if best_matrix is not None:
        logger.info("Selected alignment: ECC-refined from %s (ECC=%.4f)", best_source, best_ecc)
        matrix = best_matrix
    else:
        logger.warning("All ECC refinements failed; keeping RANSAC estimate")

    return GlobalAlignment(
        matrix=matrix,
        n_inliers=n_inliers,
        n_matches=len(good),
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
        mon_u8, init.astype(np.float32), (rw, rh),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )
    valid = (warped_mon > 0).astype(np.uint8)
    if valid.sum() < 1000:
        logger.warning(
            "ECC skipped: pre-warped mon has only %d valid pixels (need >1000)",
            int(valid.sum()),
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


def apply_global_alignment(
    monitored: GdalRasterImage,
    reference: GdalRasterImage,
    mask: Optional[GdalRasterImage],
    out_dir: Path,
) -> tuple[
    GdalRasterImage,
    GdalRasterImage,
    Optional[GdalRasterImage],
    GlobalAlignment,
]:
    """Detect the homography, apply to monitored (and mask), render onto ref's canvas.

    Returns (aligned_mon, ref_passthrough, aligned_mask, alignment_info). The
    new rasters are written to `out_dir` so the rest of the pipeline can
    operate on them as if they were the originals.
    """
    mon_arr = monitored.array
    ref_arr = reference.array

    prior = _prior_from_georefs(monitored, reference)
    if prior is not None:
        logger.info(
            "Geotransform prior available: %s", _decompose(prior),
        )
    else:
        logger.info("No geotransform prior (CRS mismatch or unreferenced)")

    alignment = detect_global_alignment(mon_arr, ref_arr, prior=prior)

    # The homography maps mon pixel coords → ref pixel coords. The warped mon
    # is rendered onto ref's canvas and saved with ref's geotransform — both
    # outputs then share a pixel grid for direct overlay/comparison.
    rh, rw = ref_arr.shape
    warp_m = alignment.matrix

    border_mon = (
        float(monitored.no_data_value)
        if monitored.no_data_value is not None
        else 0.0
    )
    aligned_mon = cv2.warpPerspective(
        mon_arr.astype(np.float32),
        warp_m,
        (rw, rh),
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
        reference.x_min,
        reference.y_max,
        reference.x_res,
        reference.y_res,
        reference.projection,
        monitored.no_data_value,
    )
    _write_geotiff(
        ref_out,
        ref_arr,
        reference.x_min,
        reference.y_max,
        reference.x_res,
        reference.y_res,
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
            cand_matrix.astype(np.float32),
            (rw, rh),
            flags=cv2.INTER_LINEAR,
            borderValue=border_mon,
        ).astype(mon_arr.dtype)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", cand_name).strip("_")
        cand_path = out_dir / f"{mon_stem}_global_aligned__{safe}_ecc{cand_ecc:.3f}{mon_suffix}"
        _write_geotiff(
            cand_path,
            cand_warped,
            reference.x_min,
            reference.y_max,
            reference.x_res,
            reference.y_res,
            reference.projection,
            monitored.no_data_value,
        )
        logger.info("Wrote alternative: %s", cand_path.name)

    aligned_mask = None
    if mask is not None:
        warped_mask = cv2.warpPerspective(
            mask.array.astype(np.uint8),
            warp_m,
            (rw, rh),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        mask_stem = Path(mask.file_name).stem
        mask_suffix = Path(mask.file_name).suffix or ".tif"
        mask_out = out_dir / f"{mask_stem}_global_aligned{mask_suffix}"
        _write_geotiff(
            mask_out,
            warped_mask,
            reference.x_min,
            reference.y_max,
            reference.x_res,
            reference.y_res,
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
