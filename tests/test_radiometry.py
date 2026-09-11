#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the shared radiometric stretch."""

import numpy as np

from karios.core.radiometry import to_uint8


def _texture(size=64, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, size=(size, size)).astype(np.float32)


def _spread(arr):
    """Dynamic range actually occupied by the bulk of the image."""
    return float(np.percentile(arr, 98) - np.percentile(arr, 2))


def test_uint8_input_is_returned_untouched():
    """Already-8-bit data must not be re-stretched."""
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)

    assert to_uint8(arr) is arr


def test_float_zero_to_one_uses_the_full_range():
    """A well-formed float raster must not be collapsed into a few DN values."""
    out = to_uint8(_texture())

    assert out.dtype == np.uint8
    assert _spread(out) > 200


def test_a_single_outlier_does_not_collapse_the_stretch():
    """One extreme pixel must not compress the rest of the image."""
    arr = _texture()
    arr[0, 0] = 50.0

    assert _spread(to_uint8(arr)) > 200


def test_a_single_infinity_does_not_collapse_the_stretch():
    """+Inf must be excluded from the statistics, not propagated through them."""
    arr = _texture()
    arr[3, 3] = np.inf

    assert _spread(to_uint8(arr)) > 200


def test_nan_is_excluded_from_the_statistics():
    """NaN must not leak into the percentile computation."""
    arr = _texture()
    arr[5, 5] = np.nan

    assert _spread(to_uint8(arr)) > 200


def test_all_non_finite_gives_zeros():
    """An unusable array degrades to black rather than raising."""
    arr = np.full((8, 8), np.nan, dtype=np.float32)

    assert np.array_equal(to_uint8(arr), np.zeros((8, 8), np.uint8))


def test_flat_image_gives_zeros():
    """A constant array has no range to stretch, and must not divide by zero."""
    out = to_uint8(np.full((8, 8), 3.5, dtype=np.float32))

    assert out.dtype == np.uint8
    assert np.array_equal(out, np.zeros((8, 8), np.uint8))


def test_single_finite_pixel_gives_zeros():
    """One usable pixel cannot define a range either."""
    arr = np.full((8, 8), np.inf, dtype=np.float32)
    arr[0, 0] = 0.5

    assert np.array_equal(to_uint8(arr), np.zeros((8, 8), np.uint8))


def test_narrower_percentiles_saturate_more():
    """The percentile window is honoured rather than ignored."""
    arr = _texture()

    tight = to_uint8(arr, percentiles=(40.0, 60.0))
    wide = to_uint8(arr, percentiles=(2.0, 98.0))

    assert tight.std() > wide.std()


def test_integer_input_is_stretched_too():
    """Non-uint8 integers are still converted, just not by the caller's choice."""
    arr = (np.arange(4096, dtype=np.uint16) * 16).reshape(64, 64)

    out = to_uint8(arr)

    assert out.dtype == np.uint8
    assert _spread(out) > 200
