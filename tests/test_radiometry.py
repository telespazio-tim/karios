#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the shared radiometric stretch."""

import numpy as np

from karios.core.radiometry import laplacian_to_uint8, to_uint8


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


def _laplacian_like_response(size=64, seed=0, scale=1000.0):
    """Signed response with the same shape a real cv2.Laplacian(CV_32F) gives:
    mostly small, a few large-magnitude outliers in both directions."""
    rng = np.random.default_rng(seed)
    small = rng.normal(0, 1, size=(size, size))
    small[0, 0] = scale
    small[1, 1] = -scale
    return small.astype(np.float32)


def test_laplacian_zero_response_is_mid_gray():
    """A flat area (zero response) must land at 128, not 0."""
    out = laplacian_to_uint8(np.zeros((8, 8), dtype=np.float32))

    assert np.array_equal(out, np.full((8, 8), 128, np.uint8))


def test_laplacian_response_keeps_its_sign():
    """A negative response must land below 128, a positive one above it -
    CV_8U instead clips every negative response to 0."""
    response = np.array([[-10.0, 10.0]], dtype=np.float32)

    out = laplacian_to_uint8(response)

    assert out[0, 0] < 128 < out[0, 1]


def test_laplacian_outlier_does_not_binarize_the_rest():
    """At power=0 (the non-binarized shape), a couple of extreme responses
    must not compress everything else onto 0/255 - the failure mode measured
    on real imagery with CV_8U (~95%+ of pixels at the two extremes)."""
    response = _laplacian_like_response()

    out = laplacian_to_uint8(response, power=0.0)

    near_extreme = np.count_nonzero((out <= 2) | (out >= 253))
    assert near_extreme / out.size < 0.05


def test_laplacian_default_power_is_near_binary():
    """The default (power=1) deliberately reproduces the old near-binary
    behaviour's cross-sensor matching robustness."""
    response = _laplacian_like_response()

    out = laplacian_to_uint8(response)

    near_extreme = np.count_nonzero((out <= 2) | (out >= 253))
    assert near_extreme / out.size > 0.9


def test_laplacian_is_monotonic():
    """The mapping is arcsinh (strictly increasing) composed with a positive
    linear rescale, so it must never decrease as the response increases -
    checked across a wide, log-spaced range of magnitudes so both the
    typical and the extreme regime are covered in one array."""
    magnitudes = np.geomspace(0.1, 10000.0, 50)
    response = np.concatenate([-magnitudes[::-1], [0.0], magnitudes]).astype(np.float32)
    response = response.reshape(1, -1)

    out = laplacian_to_uint8(response)

    assert np.all(np.diff(out[0].astype(int)) >= 0)


def test_laplacian_weak_texture_stays_separable():
    """At power=0, low-contrast regions must not be crushed toward a single
    value - a percentile-bound sigmoid does exactly that when the tile also
    contains much stronger edges elsewhere, starving goodFeaturesToTrack of
    anything to find in the weak region."""
    response = np.array([[0.05, 0.1, 0.15, 0.2]], dtype=np.float32)

    out = laplacian_to_uint8(response, power=0.0)

    assert len(set(out[0].tolist())) == 4


def test_laplacian_narrower_percentile_saturates_more():
    """At power=0 (so the shaping step doesn't itself already saturate
    everything), `percentile` calibrates contrast the same way `to_uint8`'s
    percentiles calibrate the raw-image stretch - a tighter window must use
    more of the range, not less."""
    response = _laplacian_like_response()

    tight = laplacian_to_uint8(response, percentile=60.0, power=0.0)
    wide = laplacian_to_uint8(response, percentile=99.9, power=0.0)

    assert tight.std() > wide.std()


def test_laplacian_output_is_scale_invariant():
    """Multiplying the whole response by a constant must not change the
    output at all - the bug an unnormalized fixed gain on the asinh output
    had: contrast depended on each tile's own absolute magnitude (which
    varies tile to tile and sensor to sensor), not just the shape of its
    response distribution."""
    response = _laplacian_like_response()

    out_small = laplacian_to_uint8(response * 0.001)
    out_large = laplacian_to_uint8(response * 1000.0)

    assert np.array_equal(out_small, out_large)


def test_laplacian_sparse_edges_on_a_mostly_flat_tile_are_not_discarded():
    """A tile that is flat almost everywhere, with a small sharp feature,
    must still saturate that feature - not fall back to uniform gray just
    because the response is zero at the default 98th percentile."""
    response = np.zeros((64, 64), dtype=np.float32)
    response[10, 10] = 5.0
    response[10, 11] = -5.0

    out = laplacian_to_uint8(response)

    assert out[10, 10] > 128
    assert out[10, 11] < 128


def test_laplacian_uniform_nonzero_response_does_not_divide_by_zero():
    """A perfectly uniform response has itself as the median magnitude
    (ratio 1), so it must produce a finite, sensible constant rather than
    raising a division error."""
    out = laplacian_to_uint8(np.full((8, 8), 5.0, dtype=np.float32))

    assert out.dtype == np.uint8
    value = int(out[0, 0])
    assert np.array_equal(out, np.full((8, 8), value, np.uint8))
    assert value > 128
