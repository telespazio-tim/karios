#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the coarse-to-fine matching mode."""

import cv2
import numpy as np
import pytest

from karios.core.configuration import KLTConfiguration
from karios.core.errors import ConfigurationError
from karios.matcher.coarse_to_fine import coarse_to_fine_tracker
from karios.matcher.klt import KLT, klt_tracker


def _conf(**overrides):
    base = dict(
        minDistance=10,
        blocksize=15,
        maxCorners=200000,
        matching_winsize=25,
        qualityLevel=0.1,
        xStart=0,
        tile_size=20000,
        laplacian_kernel_size=7,
        outliers_filtering=False,
        maxLevel=3,
    )
    base.update(overrides)
    return KLTConfiguration(**base)


def _shifted_pair(size=600, shift=12):
    """Textured reference and a copy displaced by `shift` px in both axes."""
    rng = np.random.default_rng(3)
    ref = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 2.0)
    ref = ((ref - ref.min()) / (ref.max() - ref.min()) * 255).astype(np.uint8)
    mon = np.roll(np.roll(ref, shift, axis=0), shift, axis=1)
    return ref, mon


def _border_fraction(points, size, band=40):
    """Share of key points lying within `band` px of the image edge."""
    x, y = points["x0"].to_numpy(), points["y0"].to_numpy()
    dist = np.minimum.reduce([x, y, size - 1 - x, size - 1 - y])
    return int((dist < band).sum())


def test_matcher_defaults_to_klt():
    """The coarse-to-fine path is opt-in, like large shift detection."""
    assert KLT(_conf())._coarse_to_fine is False


def test_auto_kernel_size_with_coarse_to_fine_is_rejected():
    """The auto-ksize search assumes single-resolution matching."""
    with pytest.raises(ConfigurationError, match="laplacian_kernel_size"):
        KLT(_conf(laplacian_kernel_size="auto"), coarse_to_fine=True)


def test_auto_kernel_size_is_fine_without_coarse_to_fine():
    """The rejection must not affect the existing matcher."""
    assert KLT(_conf(laplacian_kernel_size="auto")) is not None


def test_coarse_to_fine_tracker_honours_the_tracker_contract():
    """Same return shape as klt_tracker, so downstream code is unaffected."""
    size = 600
    ref, mon = _shifted_pair(size)
    mask = np.ones((size, size), np.uint8)

    result = coarse_to_fine_tracker(ref, mon, mask, _conf(), mon_ksize=7, ref_ksize=7)

    assert result is not None
    points, initial_count = result
    assert list(points.columns) == ["x0", "y0", "dx", "dy", "score"]
    assert initial_count >= len(points)
    assert (points["score"] <= 1).all()


def test_coarse_to_fine_recovers_border_points_lost_by_klt():
    """The pyramid must not erode matching inward from the image edges."""
    size, shift = 600, 12
    ref, mon = _shifted_pair(size, shift)
    mask = np.ones((size, size), np.uint8)
    conf = _conf()

    ref_lap = cv2.Laplacian(ref, cv2.CV_8U, ksize=7)
    mon_lap = cv2.Laplacian(mon, cv2.CV_8U, ksize=7)
    klt_points, _ = klt_tracker(ref_lap, mon_lap, mask, conf)
    ctf_points, _ = coarse_to_fine_tracker(ref, mon, mask, conf, mon_ksize=7, ref_ksize=7)

    assert _border_fraction(ctf_points, size) > _border_fraction(klt_points, size)


def test_coarse_to_fine_recovers_the_true_displacement():
    """Recovered key points must describe the real shift, not merely be numerous."""
    size, shift = 600, 12
    ref, mon = _shifted_pair(size, shift)
    mask = np.ones((size, size), np.uint8)

    points, _ = coarse_to_fine_tracker(ref, mon, mask, _conf(), mon_ksize=7, ref_ksize=7)

    assert points["dx"].median() == pytest.approx(shift, abs=0.5)
    assert points["dy"].median() == pytest.approx(shift, abs=0.5)


def _runtime(**overrides):
    from karios.api.config import RuntimeConfiguration

    base = dict(
        output_directory="/tmp/karios-test",
        gen_kp_mask=False,
        gen_delta_raster=False,
        generate_kp_chips=False,
        enable_large_shift_detection=False,
    )
    base.update(overrides)
    return RuntimeConfiguration(**base)


def test_large_shift_and_coarse_to_fine_are_mutually_exclusive():
    """Both correct a coarse displacement; running both is contradictory."""
    with pytest.raises(ConfigurationError, match="coarse-to-fine"):
        _runtime(enable_large_shift_detection=True, enable_coarse_to_fine=True)


def test_either_switch_alone_is_accepted():
    """The exclusion must not block the individual switches."""
    assert _runtime(enable_large_shift_detection=True).enable_large_shift_detection is True
    assert _runtime(enable_coarse_to_fine=True).enable_coarse_to_fine is True
