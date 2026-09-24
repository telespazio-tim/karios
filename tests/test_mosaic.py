#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the checkerboard mosaic of the monitored and reference images."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, features

from karios.api.config import RuntimeConfiguration
from karios.core.errors import ConfigurationError
from karios.report import mosaic
from karios.report.mosaic import checkerboard, generate_mosaic


AVIF_SUPPORTED = features.check("avif")


def _image(array, no_data_value=None):
    return SimpleNamespace(array=array, no_data_value=no_data_value, filepath="fake.tif")


def _read(path):
    """Decode a mosaic as a 2D uint8 array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


@pytest.fixture(name="lossless", autouse=True)
def lossless_fixture(monkeypatch):
    """Exact pixel checks below need the lossless PNG output: disable AVIF."""
    monkeypatch.setattr(mosaic.features, "check", lambda name: False)


def test_checkerboard_alternates_tiles():
    board = checkerboard((4, 6), 2)

    assert board.shape == (4, 6)
    assert not board[0, 0] and not board[1, 1]
    assert board[0, 2] and board[2, 0]
    assert not board[2, 2]
    assert board[:, 4:].tolist() == board[:, :2].tolist()


def test_mosaic_alternates_reference_and_monitored(tmp_path, monkeypatch):
    """Reference on the top left tile, monitored on its neighbours."""
    ramp = np.linspace(1, 100, 16 * 16, dtype=np.float32).reshape(16, 16)
    ref = _image(ramp)
    mon = _image(ramp[::-1].copy())

    full = generate_mosaic(mon, ref, tmp_path / "mosaic", 8)
    out = _read(full)

    assert out.shape == (16, 16)
    assert out.dtype == np.uint8
    ref_tile = out[:8, :8]
    mon_tile = out[:8, 8:]
    # Reference ramp increases downward, the flipped monitored one decreases
    assert ref_tile[-1, 0] > ref_tile[0, 0]
    assert mon_tile[-1, 0] < mon_tile[0, 0]


def test_mosaic_uses_the_overview_contrast(tmp_path):
    """Pixels outside the 0.5-99.5% cut saturate, like in the overview plot."""
    array = np.random.default_rng(0).uniform(1, 100, size=(40, 40)).astype(np.float32)
    array[0, 0] = 1e6
    array[0, 1] = -1e6
    img = _image(array)

    full = generate_mosaic(img, img, tmp_path / "mosaic", 16)
    out = _read(full)

    assert out[0, 0] == 255
    assert out[0, 1] == 0
    # The outliers do not compress the rest of the image
    assert np.percentile(out, 98) - np.percentile(out, 2) > 200


def test_mosaic_hides_invalid_pixels(tmp_path, monkeypatch):
    """No-data, --no-value and non-finite pixels are black; the mask hides monitored only."""
    ref_array = np.full((4, 4), 50, dtype=np.float32)
    ref_array[0, 0] = -9999  # no-data, reference tile
    ref_array[1, 1] = np.nan  # reference tile
    ref_array[3, 3] = 60  # keep a contrast range
    mon_array = np.full((4, 4), 50, dtype=np.float32)
    mon_array[0, 3] = 7  # --no-value, monitored tile
    mon_array[3, 1] = 60
    mask_array = np.zeros((4, 4), dtype=np.uint8)
    mask_array[2:, :2] = 1  # keep the bottom left monitored tile visible

    full = generate_mosaic(
        _image(mon_array),
        _image(ref_array, no_data_value=-9999),
        tmp_path / "mosaic",
        2,
        mask=_image(mask_array),
        no_values=[7],
    )
    out = _read(full)

    assert out[0, 0] == 0
    assert out[1, 1] == 0
    assert out[0, 3] == 0
    # Masked monitored tile
    assert out[1, 2] == 0
    # Reference tiles ignore the mask, unmasked monitored tile is visible
    assert out[3, 3] > 0
    assert out[3, 1] > 0


@pytest.mark.parametrize("avif", [True, False])
def test_full_mosaic_keeps_native_size(tmp_path, monkeypatch, avif):
    """Native mosaic is AVIF when Pillow supports it, lossless PNG otherwise."""
    if avif and not AVIF_SUPPORTED:
        pytest.skip("Pillow built without AVIF")
    monkeypatch.setattr(mosaic.features, "check", lambda name: avif and name == "avif")
    array = np.random.default_rng(0).uniform(1, 100, size=(1500, 3000)).astype(np.float32)
    img = _image(array)

    full = generate_mosaic(img, img, tmp_path / "mosaic", 16)

    assert full.name == ("mosaic.avif" if avif else "mosaic.png")
    with Image.open(full) as im:
        assert im.format == ("AVIF" if avif else "PNG")
    assert _read(full).shape == (1500, 3000)


def test_tiles_have_the_requested_size(tmp_path):
    texture = np.random.default_rng(0).uniform(1, 100, size=(300, 300)).astype(np.float32)
    ref = _image(texture)
    mon = _image(texture[::-1].copy())

    out = _read(generate_mosaic(mon, ref, tmp_path / "mosaic", 128)).astype(int)

    ref_gray = _read(generate_mosaic(ref, ref, tmp_path / "ref", 128)).astype(int)
    mon_gray = _read(generate_mosaic(mon, mon, tmp_path / "mon", 128)).astype(int)
    tiles = np.arange(300) // 128
    expected_mon = (tiles[:, np.newaxis] + tiles[np.newaxis, :]) % 2 == 1
    assert np.array_equal(out[expected_mon], mon_gray[expected_mon])
    assert np.array_equal(out[~expected_mon], ref_gray[~expected_mon])


def _runtime_configuration(tmp_path, **kwargs):
    return RuntimeConfiguration(
        output_directory=tmp_path,
        gen_kp_mask=False,
        gen_delta_raster=False,
        generate_kp_chips=False,
        enable_large_shift_detection=False,
        **kwargs,
    )


def test_mosaic_is_disabled_by_default(tmp_path):
    assert _runtime_configuration(tmp_path).mosaic_tile_size == 0


def test_negative_mosaic_tile_size_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="mosaic_tile_size"):
        _runtime_configuration(tmp_path, mosaic_tile_size=-1)
