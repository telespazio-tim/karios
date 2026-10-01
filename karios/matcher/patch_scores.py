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
"""Similarity scores of the key point patches, computed for many points at once.

For each key point, a 57x57 patch centered on it in the reference and one
centered on its matched position in the monitored image give:

- `zncc_score`: zero-mean normalized cross-correlation of their inner 43x43,
- `mutual_info_score`: Studholme's normalized mutual information, (H(X) + H(Y)) / H(X, Y),
- `mi_score`: normalized mutual information, 2 MI(X, Y) / (H(X) + H(Y)).

They reproduce the per-point functions of zncc_service and
mutual_info_service, which ran one pandas apply per score and one
np.histogram2d per point, twice for the same joint histogram: 50 s for 11k
points on a 30 m scene. Here patches are gathered by array indexing, a
chunk of points at a time, and the joint histograms are counted with one
bincount, on the very bin edges np.histogram2d would use. ZNCC is computed
in float64, where the per-point function kept float32 patches in float32:
scores of float32 images differ by about 1e-7.
"""

import logging

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

logger = logging.getLogger(__name__)

CHIP_SIZE = 57
ZNCC_RADIUS = 21  # inner patch of 2 * 21 + 1 = 43 px
MI_BINS = 32
CHUNK_POINTS = 2000  # points per chunk: two float64 57x57 patch stacks of 52 MB each


def _histogram_bins(values: NDArray) -> NDArray:
    """Bin index in [0, MI_BINS) of every value of each patch, as np.histogram2d computes it.

    `values` is (points, pixels). Each patch spans MI_BINS bins between its
    own minimum and maximum, the maximum in the last bin, and a constant patch
    spans [value - 0.5, value + 0.5]. np.histogram2d builds those edges in
    float64 whatever the samples' type (NumPy 1 promotes their scalar bounds),
    so both mutual informations, one on the raw samples and one on their
    float64 copy, see the same bins.
    """
    samples = values.astype(np.float64, copy=False)
    lo = samples.min(axis=1)
    hi = samples.max(axis=1)
    constant = lo == hi
    lo = np.where(constant, lo - 0.5, lo)
    hi = np.where(constant, hi + 0.5, hi)
    # np.linspace(lo, hi, MI_BINS + 1) computes edge k < MI_BINS as k * step + lo,
    # and puts hi itself last: the same arithmetic gives the exact edges here
    # without building or gathering them
    step = ((hi - lo) / MI_BINS)[:, None]
    lo = lo[:, None]

    # Approximate bin, then corrected against the actual edges: a value is in
    # bin k when edges[k] <= value < edges[k + 1], the last bin closed
    bins = np.clip(np.floor((samples - lo) / step), 0, MI_BINS - 1)
    bins -= (samples < bins * step + lo) & (bins > 0)
    bins += (samples >= (bins + 1) * step + lo) & (bins < MI_BINS - 1)
    bins = bins.astype(np.int64)
    return bins


def _joint_histograms(ref: NDArray, mon: NDArray) -> NDArray:
    """(points, MI_BINS, MI_BINS) joint histograms of each reference and monitored patch."""
    bins_ref = _histogram_bins(ref)
    bins_mon = _histogram_bins(mon)
    n = len(ref)
    flat = (np.arange(n)[:, None] * MI_BINS + bins_ref) * MI_BINS + bins_mon
    counts = np.bincount(flat.ravel(), minlength=n * MI_BINS * MI_BINS)
    return counts.reshape(n, MI_BINS, MI_BINS).astype(np.float64)


def _entropies(joint: NDArray, log) -> tuple[NDArray, NDArray, NDArray]:
    """H(X), H(Y) and H(X, Y) of each joint histogram, with the `log` base given."""
    pxy = joint / joint.sum(axis=(1, 2), keepdims=True)

    def entropy(p: NDArray, axes) -> NDArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * log(np.where(p > 0, p, 1.0)), 0.0)
        return -terms.sum(axis=axes)

    return entropy(pxy.sum(axis=2), 1), entropy(pxy.sum(axis=1), 1), entropy(pxy, (1, 2))


def _zncc(ref: NDArray, mon: NDArray) -> NDArray:
    """ZNCC of the inner (2 ZNCC_RADIUS + 1)^2 of each patch pair, NaN for a constant patch."""
    c, r = CHIP_SIZE // 2, ZNCC_RADIUS
    inner = (slice(None), slice(c - r, c + r + 1), slice(c - r, c + r + 1))
    p1 = ref.reshape(-1, CHIP_SIZE, CHIP_SIZE)[inner].astype(np.float64)
    p2 = mon.reshape(-1, CHIP_SIZE, CHIP_SIZE)[inner].astype(np.float64)
    std1 = p1.std(axis=(1, 2))
    std2 = p2.std(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        norm1 = (p1 - p1.mean(axis=(1, 2), keepdims=True)) / std1[:, None, None]
        norm2 = (p2 - p2.mean(axis=(1, 2), keepdims=True)) / std2[:, None, None]
        zncc = (norm1 * norm2).mean(axis=(1, 2))
    return np.where((std1 == 0) | (std2 == 0), np.nan, zncc)


def _patches(array: NDArray, x: NDArray, y: NDArray) -> NDArray:
    """(points, CHIP_SIZE ** 2) patches of `array` centered on (x, y)."""
    offsets = np.arange(CHIP_SIZE) - CHIP_SIZE // 2
    rows = (y[:, None] + offsets)[:, :, None]
    cols = (x[:, None] + offsets)[:, None, :]
    return array[rows, cols].reshape(len(x), -1)


def compute_patch_scores(df: DataFrame, monitored, reference) -> DataFrame:
    """ZNCC, Studholme's and normalized mutual information of each key point's patches.

    Args:
        df: key points with columns x0, y0 (reference position) and dx, dy
            (shift to the monitored position, rounded half to even like round()).
        monitored, reference: images with `array`, `x_size` and `y_size`.

    Returns:
        DataFrame with columns zncc_score, mutual_info_score and mi_score on
        df's index, NaN for points whose patches leave either image, or
        contain non-finite values, or are constant (where each score is
        undefined).
    """
    scores = DataFrame(
        np.nan, index=df.index, columns=["zncc_score", "mutual_info_score", "mi_score"]
    )
    if df.empty:
        return scores

    x0 = df["x0"].to_numpy().astype(np.int64)  # truncated, as int() did
    y0 = df["y0"].to_numpy().astype(np.int64)
    x1 = np.round(df["x0"].to_numpy() + df["dx"].to_numpy()).astype(np.int64)
    y1 = np.round(df["y0"].to_numpy() + df["dy"].to_numpy()).astype(np.int64)

    def within(width, height) -> NDArray:
        m = CHIP_SIZE // 2
        return (
            (x0 >= m) & (y0 >= m) & (x0 < width[0] - m) & (y0 < height[0] - m)
            & (x1 >= m) & (y1 >= m) & (x1 < width[1] - m) & (y1 < height[1] - m)
        )  # fmt: skip

    # Image sizes first: the arrays are only read when some point needs them
    inside = within((reference.x_size, monitored.x_size), (reference.y_size, monitored.y_size))
    if inside.any():
        ref_array, mon_array = reference.array, monitored.array
        inside &= within(
            (ref_array.shape[1], mon_array.shape[1]), (ref_array.shape[0], mon_array.shape[0])
        )
    if not inside.all():
        logger.warning("%d points too close to image boundaries, skipped", int((~inside).sum()))

    result = np.full((len(df), 3), np.nan)
    kept = np.flatnonzero(inside)
    for start in range(0, len(kept), CHUNK_POINTS):
        chunk = kept[start : start + CHUNK_POINTS]
        ref = _patches(ref_array, x0[chunk], y0[chunk])
        mon = _patches(mon_array, x1[chunk], y1[chunk])

        # A non-finite value in its inner patch makes ZNCC NaN by itself
        result[chunk, 0] = _zncc(ref, mon)

        # np.histogram2d refuses non-finite values: no mutual information there
        finite = np.isfinite(ref).all(axis=1) & np.isfinite(mon).all(axis=1)
        if not finite.all():
            chunk, ref, mon = chunk[finite], ref[finite], mon[finite]
        if len(chunk) == 0:
            continue

        joint = _joint_histograms(ref, mon)
        hx, hy, hxy = _entropies(joint, np.log)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[chunk, 1] = np.where(hxy == 0, np.nan, (hx + hy) / hxy)

        hx, hy, hxy = _entropies(joint, np.log2)
        denominator = hx + hy
        with np.errstate(divide="ignore", invalid="ignore"):
            result[chunk, 2] = np.where(
                denominator == 0, np.nan, 2.0 * (hx + hy - hxy) / denominator
            )

    scores[:] = result
    return scores
