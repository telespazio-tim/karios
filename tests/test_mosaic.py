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
from karios.report.mosaic import checkerboard, generate_mosaic, generate_overlay


AVIF_SUPPORTED = features.check("avif")
DEFAULT_CHROMA = mosaic.MOSAIC_CHROMA


def _image(array, no_data_value=None):
    return SimpleNamespace(array=array, no_data_value=no_data_value, filepath="fake.tif")


def _gray(im):
    """Gray level of a mosaic, untinted by the fixture below: every channel holds it."""
    return np.asarray(im.convert("RGB")).max(axis=2)


def _read(path):
    """Decode a mosaic as a 2D uint8 array of gray levels."""
    with Image.open(path) as im:
        return _gray(im)


def _read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


@pytest.fixture(name="lossless", autouse=True)
def lossless_fixture(monkeypatch):
    """Exact pixel checks below need the lossless PNG output: disable AVIF."""
    monkeypatch.setattr(mosaic.features, "check", lambda name: False)


@pytest.fixture(name="untinted", autouse=True)
def untinted_fixture(monkeypatch):
    """Gray level checks below need an untinted checkerboard; tint tests restore the chroma."""
    monkeypatch.setattr(mosaic, "MOSAIC_CHROMA", 0)


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


def test_oklab_matches_the_reference_values():
    """Values from https://bottosson.github.io/posts/oklab/"""
    assert np.allclose(mosaic.srgb_to_oklab([1, 1, 1]), [1, 0, 0], atol=1e-4)
    assert np.allclose(mosaic.srgb_to_oklab([1, 0, 0]), [0.62796, 0.22486, 0.12585], atol=1e-4)
    rgb = np.random.default_rng(0).uniform(size=(100, 3))
    assert np.allclose(mosaic.oklab_to_srgb(mosaic.srgb_to_oklab(rgb)), rgb)


@pytest.mark.parametrize("hue", [mosaic.MONITORED_HUE, mosaic.REFERENCE_HUE])
def test_tint_keeps_the_gray_lightness(monkeypatch, hue):
    """Each tint is as light as its gray level, at `hue` and up to MOSAIC_CHROMA."""
    monkeypatch.setattr(mosaic, "MOSAIC_CHROMA", DEFAULT_CHROMA)

    lab = mosaic.srgb_to_oklab(mosaic._tint(hue) / 255)
    gray = mosaic.srgb_to_oklab(np.repeat(np.arange(256)[:, np.newaxis] / 255, 3, axis=1))
    chroma = np.hypot(lab[:, 1], lab[:, 2])

    # Tolerances cover the 8 bit rounding, which moves a and b by up to about
    # 0.003: the lower the chroma, the more hue that is. The steep low end of the
    # sRGB curve makes the darkest levels coarser, in lightness and chroma.
    rounding = 0.003
    assert np.allclose(lab[8:, 0], gray[8:, 0], atol=0.005)
    assert np.allclose(lab[:, 0], gray[:, 0], atol=0.01)
    assert (chroma[8:] <= DEFAULT_CHROMA + rounding).all()
    assert (chroma <= DEFAULT_CHROMA + 0.01).all()
    # Mid grays get the full chroma, black and white none
    assert np.allclose(chroma[64:200], DEFAULT_CHROMA, atol=rounding)
    assert chroma[0] < rounding and chroma[255] < rounding
    hue_error = (np.degrees(np.arctan2(lab[32:200, 2], lab[32:200, 1])) - hue + 180) % 360 - 180
    assert (np.abs(hue_error) < np.degrees(np.arcsin(rounding / DEFAULT_CHROMA))).all()


def test_mosaic_tints_monitored_in_red_and_reference_in_blue(tmp_path, monkeypatch):
    monkeypatch.setattr(mosaic, "MOSAIC_CHROMA", DEFAULT_CHROMA)
    img = _image(np.random.default_rng(0).uniform(1, 100, size=(16, 16)).astype(np.float32))

    out = _read_rgb(generate_mosaic(img, img, tmp_path / "mosaic", 8))

    gray = mosaic._to_gray(img, None)
    assert np.array_equal(out[:8, :8], mosaic._tint(mosaic.REFERENCE_HUE)[gray[:8, :8]])
    assert np.array_equal(out[:8, 8:], mosaic._tint(mosaic.MONITORED_HUE)[gray[:8, 8:]])


def test_mosaic_equalizes_the_histogram(tmp_path):
    """A skewed distribution spreads evenly over the gray levels, outliers included."""
    array = np.random.default_rng(0).exponential(10, size=(64, 64)).astype(np.float32) + 1
    array[0, 0] = 1e6
    array[0, 1] = -1e6
    img = _image(array)

    out = _read(generate_mosaic(img, img, tmp_path / "mosaic", 16))

    assert out[0, 0] == 255
    assert out[0, 1] == 0
    # Flat histogram: each quarter of the gray range holds about a quarter of the pixels
    quarters = np.histogram(out, bins=4, range=(0, 256))[0] / out.size
    assert np.allclose(quarters, 0.25, atol=0.02)


def test_mosaic_gives_both_images_the_same_contrast(tmp_path):
    """Equalization ignores the radiometric range: a rescaled image looks the same."""
    ref_array = np.random.default_rng(0).uniform(1, 100, size=(32, 32)).astype(np.float32)
    mon_array = ref_array * 250 + 1000

    out = _read(generate_mosaic(_image(mon_array), _image(ref_array), tmp_path / "mosaic", 8))
    ref_gray = _read(generate_mosaic(_image(ref_array), _image(ref_array), tmp_path / "ref", 8))

    assert np.array_equal(out, ref_gray)


def test_mosaic_leaves_zero_fill_out_of_the_histogram(tmp_path):
    """Zero fill stays black and does not take gray levels from the scene."""
    scene = np.random.default_rng(0).uniform(1, 100, size=(32, 32)).astype(np.float32)
    filled = np.zeros((32, 64), dtype=np.float32)
    filled[:, 32:] = scene

    out = _read(generate_mosaic(_image(filled), _image(filled), tmp_path / "filled", 8))
    alone = _read(generate_mosaic(_image(scene), _image(scene), tmp_path / "scene", 8))

    assert not out[:, :32].any()
    assert np.array_equal(out[:, 32:], alone)


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


def test_overlay_puts_monitored_in_red_and_reference_in_green_and_blue(tmp_path):
    rng = np.random.default_rng(0)
    ref_array = rng.uniform(1, 100, size=(32, 32)).astype(np.float32)
    mon_array = rng.uniform(1, 100, size=(32, 32)).astype(np.float32)
    ref, mon = _image(ref_array), _image(mon_array)

    out = _read_rgb(generate_overlay(mon, ref, tmp_path / "overlay"))

    mon_gray = _read(generate_mosaic(mon, mon, tmp_path / "mon", 8))
    ref_gray = _read(generate_mosaic(ref, ref, tmp_path / "ref", 8))
    assert np.array_equal(out[..., 0], mon_gray)
    assert np.array_equal(out[..., 1], ref_gray)
    assert np.array_equal(out[..., 2], ref_gray)


def test_overlay_of_identical_images_is_gray(tmp_path):
    img = _image(np.random.default_rng(0).uniform(1, 100, size=(32, 32)).astype(np.float32))

    out = _read_rgb(generate_overlay(img, img, tmp_path / "overlay"))

    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()


def test_overlay_hides_masked_pixels_of_the_monitored_image_only(tmp_path):
    img = _image(np.random.default_rng(0).uniform(1, 100, size=(8, 8)).astype(np.float32))
    mask_array = np.ones((8, 8), dtype=np.uint8)
    mask_array[:4] = 0

    out = _read_rgb(generate_overlay(img, img, tmp_path / "overlay", mask=_image(mask_array)))

    assert not out[:4, :, 0].any()
    assert out[:4, :, 1:].any()
    assert out[4:, :, 0].any()


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


def test_overlay_is_disabled_by_default(tmp_path):
    assert not _runtime_configuration(tmp_path).generate_overlay


def test_negative_mosaic_tile_size_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="mosaic_tile_size"):
        _runtime_configuration(tmp_path, mosaic_tile_size=-1)
