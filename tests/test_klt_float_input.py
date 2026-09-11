#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matching must survive the pixel values a float raster can carry.

These drive the real tracker rather than mocking OpenCV: the failure being
guarded against is entirely about what the stretch hands to `cv2.Laplacian`, so
a mocked Laplacian would not see it.
"""

import cv2
import numpy as np
import pytest

from karios.core.configuration import KLTConfiguration
from karios.matcher.klt import _stretch, klt_tracker


def _conf():
    return KLTConfiguration(
        minDistance=10,
        blocksize=15,
        maxCorners=200000,
        matching_winsize=25,
        qualityLevel=0.1,
        xStart=0,
        tile_size=20000,
        laplacian_kernel_size=7,
        outliers_filtering=False,
        maxLevel=1,
    )


def _pair(size=400, shift=3):
    """A textured uint16 reference and a copy displaced by `shift` px."""
    rng = np.random.default_rng(1)
    ref = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 2.0)
    ref = ((ref - ref.min()) / (ref.max() - ref.min()) * 4000).astype(np.uint16)
    mon = np.roll(np.roll(ref, shift, axis=0), shift, axis=1)
    return ref, mon


def _match(ref, mon):
    """Run the real KLT chain, returning the number of key points kept."""
    conf = _conf()
    mask = np.ones(ref.shape, np.uint8)
    ref_lap = cv2.Laplacian(_stretch(ref), cv2.CV_8U, ksize=7)
    mon_lap = cv2.Laplacian(_stretch(mon), cv2.CV_8U, ksize=7)
    result = klt_tracker(ref_lap, mon_lap, mask, conf)
    return 0 if result is None else len(result[0])


@pytest.fixture(name="baseline")
def baseline_fixture():
    ref, mon = _pair()
    count = _match(ref, mon)
    assert count > 0, "the integer baseline itself found nothing; fixture is broken"
    return count


def _as_float(arr, scale=4000.0):
    return arr.astype(np.float32) / scale


def test_float_input_matches_as_well_as_the_integer_original(baseline):
    """Rescaling to float must not cost key points."""
    ref, mon = _pair()

    count = _match(_as_float(ref), _as_float(mon))

    assert count >= 0.9 * baseline


def test_one_outlier_pixel_does_not_stop_matching(baseline):
    """A single wild value used to collapse the stretch and yield nothing."""
    ref, mon = _pair()
    ref_f = _as_float(ref)
    ref_f[0, 0] = 50.0

    count = _match(ref_f, _as_float(mon))

    assert count >= 0.9 * baseline


def test_one_infinite_pixel_does_not_stop_matching(baseline):
    """+Inf survives nanmax, so it used to drive the whole image to zero."""
    ref, mon = _pair()
    ref_f = _as_float(ref)
    ref_f[5, 5] = np.inf

    count = _match(ref_f, _as_float(mon))

    assert count >= 0.9 * baseline


def test_sentinel_no_data_does_not_stop_matching(baseline):
    """A large negative fill value is as damaging as a large positive one."""
    ref, mon = _pair()
    ref_f = _as_float(ref)
    ref_f[:5, :5] = -9999.0

    count = _match(ref_f, _as_float(mon))

    assert count >= 0.9 * baseline


def test_integer_input_keeps_its_existing_conversion():
    """Integer rasters are deliberately left on the previous behaviour."""
    ref, _ = _pair()

    stretched = _stretch(ref)

    expected = ((ref - ref.min()) / (ref.max() - ref.min()) * 255).astype(np.uint8)
    assert np.array_equal(stretched, expected)
