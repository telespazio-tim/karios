#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the SIFT keypoint limit of the global alignment."""

import logging
import re

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from osgeo import gdal

from karios.cli import commands
from karios.core.image import GdalRasterImage
from karios.matcher import global_align
from karios.matcher.global_align import detect_global_alignment


@pytest.fixture(name="shifted_pair", scope="module")
def shifted_pair_fixture():
    """(mon, ref) crops of one texture, mon = ref moved by (+5, +3) px."""
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (600, 600)).astype(np.float32), (0, 0), 2)
    ref = texture[50:562, 50:562]
    mon = texture[53:565, 55:567]
    return mon, ref


def _keypoint_counts(caplog) -> tuple[int, int]:
    match = re.search(r"Keypoints detected: mon=(\d+)  ref=(\d+)", caplog.text)
    return int(match.group(1)), int(match.group(2))


def test_sift_nfeatures_limits_keypoints_per_image(shifted_pair, caplog):
    mon, ref = shifted_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)

    alignment = detect_global_alignment(mon, ref, sift_nfeatures=300)

    # OpenCV also keeps the keypoints tied with the weakest one kept
    assert all(300 <= count <= 310 for count in _keypoint_counts(caplog))
    # The strongest keypoints are enough to recover the shift
    assert alignment.matrix[0, 2] == pytest.approx(5, abs=0.5)
    assert alignment.matrix[1, 2] == pytest.approx(3, abs=0.5)


def test_sift_nfeatures_defaults_to_unlimited(shifted_pair, caplog):
    mon, ref = shifted_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)

    detect_global_alignment(mon, ref)

    assert global_align.SIFT_NFEATURES == 0
    assert min(_keypoint_counts(caplog)) > 300


def test_negative_sift_nfeatures_is_rejected(shifted_pair):
    mon, ref = shifted_pair

    with pytest.raises(ValueError, match="sift_nfeatures"):
        detect_global_alignment(mon, ref, sift_nfeatures=-1)


def _run_align(tmp_path, monkeypatch, *options):
    """Invoke `karios align` on dummy files, capturing apply_global_alignment kwargs."""
    calls = []

    def fake_apply(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop after the call")

    monkeypatch.setattr(commands, "GdalRasterImage", lambda path: path)
    monkeypatch.setattr(commands, "apply_global_alignment", fake_apply)
    mon, ref = tmp_path / "mon.tif", tmp_path / "ref.tif"
    mon.touch()
    ref.touch()
    result = CliRunner().invoke(
        commands.cli,
        ["align", str(mon), str(ref), "--out", str(tmp_path / "out"), "--no-log-file", *options],
    )
    return result, calls


def test_cli_passes_sift_nfeatures(tmp_path, monkeypatch):
    result, calls = _run_align(tmp_path, monkeypatch, "--sift-nfeatures", "5000")

    assert result.exit_code == 0, result.output
    assert calls == [{"sift_nfeatures": 5000}]


def test_cli_sift_nfeatures_defaults_to_unlimited(tmp_path, monkeypatch):
    _, calls = _run_align(tmp_path, monkeypatch)

    assert calls == [{"sift_nfeatures": 0}]


def test_cli_rejects_negative_sift_nfeatures(tmp_path, monkeypatch):
    result, calls = _run_align(tmp_path, monkeypatch, "--sift-nfeatures", "-1")

    assert result.exit_code != 0
    assert "sift-nfeatures" in result.output
    assert not calls


@pytest.fixture(name="fine_pair", scope="module")
def fine_pair_fixture():
    """(mon, ref, truth): mon is 4x finer than ref and covers ref[150:350, 180:380].

    `truth` is the mon → ref homography, scale 1/4 and the (180, 150) offset.
    """
    rng = np.random.default_rng(1)
    ref = cv2.GaussianBlur(rng.uniform(0, 255, (600, 600)).astype(np.float32), (0, 0), 3)
    ref = cv2.normalize(ref, None, 10, 250, cv2.NORM_MINMAX)
    mon = cv2.resize(ref[150:350, 180:380], (800, 800), interpolation=cv2.INTER_CUBIC)
    return mon, ref, _truth(180, 150, 4)


def _truth(x0, y0, factor):
    """mon → ref homography of a mon `factor` times finer than ref, from ref pixel (x0, y0).

    Maps OpenCV pixel centers, as cv2.resize samples them.
    """
    offset = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
    return offset @ global_align._pixel_scale(1 / factor, 1 / factor)


def _offset(truth, dx, dy, scale=1.0):
    """`truth` with its translation moved by (dx, dy) ref px and its scale multiplied."""
    prior = truth.copy()
    prior[:2, :2] *= scale
    prior[0, 2] += dx
    prior[1, 2] += dy
    return prior


def _center_error(matrix, truth, size=800):
    """Distance in ref px between where `matrix` and `truth` put mon's center and corners."""
    points = np.array([[size / 2, size / 2], [0, 0], [size, 0], [size, size], [0, size]], float)
    return float(
        np.abs(
            global_align._footprint_points(matrix, points)
            - global_align._footprint_points(truth, points)
        ).max()
    )


@pytest.mark.parametrize(
    "dx, dy, scale",
    [(40, -30, 1.0), (-60, 45, 1.0), (30, 20, 1.15)],
    ids=["shift", "larger-shift", "shift-and-scale"],
)
def test_prior_far_off_is_corrected(fine_pair, dx, dy, scale):
    """A georeferencing tens of ref px off, even with a wrong scale, is corrected."""
    mon, ref, truth = fine_pair

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, dx, dy, scale))

    assert _center_error(alignment.matrix, truth) < 0.5


def test_translation_search_recovers_the_shift(fine_pair, caplog):
    mon, ref, truth = fine_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)

    detect_global_alignment(mon, ref, prior=_offset(truth, -60, 45))

    match = re.search(r"shift from prior=\(([-+\d.]+), ([-+\d.]+)\)", caplog.text)
    assert float(match.group(1)) == pytest.approx(60, abs=1)
    assert float(match.group(2)) == pytest.approx(-45, abs=1)


def test_alignment_without_sift_falls_back_on_the_other_starts(fine_pair, monkeypatch):
    """SIFT failing is not fatal with a prior: the translation search still gives the answer."""
    mon, ref, truth = fine_pair

    def no_sift(*args, **kwargs):
        raise RuntimeError("Too few good matches")

    monkeypatch.setattr(global_align, "_sift_homography", no_sift)

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, 40, -30))

    assert _center_error(alignment.matrix, truth) < 0.5
    assert alignment.n_matches == 0


def test_implausible_sift_estimate_is_rejected(fine_pair, monkeypatch, caplog):
    """A mirrored RANSAC estimate is dropped even if its ECC run converges."""
    mon, ref, truth = fine_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)
    mirror = np.array([[-1.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    monkeypatch.setattr(global_align, "_sift_homography", lambda *args, **kwargs: (mirror, 50, 60))

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, 40, -30))

    assert _center_error(alignment.matrix, truth) < 0.5
    assert all(name != "RANSAC" for name, _, _ in alignment.candidates)


@pytest.mark.parametrize(
    "linear, translation, reason",
    [
        ([[-1, 0], [0, 1]], (0, 0), "mirrors"),
        ([[1.8, 0], [0, 1.8]], (0, 0), "scales"),
        ([[0.5, 0], [0, 0.5]], (0, 0), "scales"),
        ([[1.4, 0], [0, 1.0]], (0, 0), "anisotropy"),
        ([[0.7, -0.7], [0.7, 0.7]], (0, 0), "rotates"),
        ([[1, 0], [0, 1]], (500, 0), "moves"),
    ],
)
def test_plausibility_limits(fine_pair, linear, translation, reason):
    """Departures from the prior beyond the limits are named; the prior itself passes."""
    mon, ref, truth = fine_pair
    frame = global_align._work_frame(
        global_align._preprocess(mon), mon > 0, global_align._preprocess(ref), truth
    )
    # Apply the change around mon's center, in ref px
    center = global_align._footprint_points(truth, np.array([[400.0, 400.0]]))[0]
    change = np.eye(3)
    change[:2, :2] = linear
    change[:2, 2] = center - np.array(linear) @ center + np.array(translation)

    assert global_align._plausibility(truth, truth, frame) is None
    assert reason in global_align._plausibility(change @ truth, truth, frame)


def test_prior_outside_ref_is_ignored(shifted_pair, caplog):
    """A prior pointing away from ref falls back to matching the whole images."""
    mon, ref = shifted_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)
    prior = np.array([[1.0, 0.0, 5000.0], [0.0, 1.0, 5000.0], [0.0, 0.0, 1.0]])

    alignment = detect_global_alignment(mon, ref, prior=prior)

    assert "prior ignored" in caplog.text
    assert alignment.matrix[0, 2] == pytest.approx(5, abs=0.5)
    assert alignment.matrix[1, 2] == pytest.approx(3, abs=0.5)


def test_small_fine_footprint_in_a_large_reference():
    """The PhiSat case: mon 7x finer, 100 ref px wide in a 1500 px ref, with detail ref lacks.

    SIFT on the whole images matches mon's fine detail with ref keypoints far
    outside its footprint; the previous pipeline landed 32 px off at the
    corners. At a common resolution around the prior footprint the translation
    is recovered and only the scale of such a small footprint stays uncertain.
    """
    rng = np.random.default_rng(4)
    ref = cv2.GaussianBlur(rng.uniform(0, 255, (1500, 1500)).astype(np.float32), (0, 0), 3)
    ref = cv2.normalize(ref, None, 10, 250, cv2.NORM_MINMAX)
    x0, y0 = 717, 700
    mon = cv2.resize(ref[y0 : y0 + 100, x0 : x0 + 100], (700, 700), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.normal(0, 1, mon.shape).astype(np.float32), (0, 0), 1.5)
    mon = np.clip(mon + fine * 40 / fine.std(), 1, 255)
    truth = _truth(x0, y0, 7)

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, -30, 25))

    center = np.array([[350.0, 350.0]])
    assert (
        np.abs(
            global_align._footprint_points(alignment.matrix, center)
            - global_align._footprint_points(truth, center)
        ).max()
        < 1
    )
    assert _center_error(alignment.matrix, truth, size=700) < 5


def test_downsampled_working_image_keeps_pixel_centers():
    """mon 7x finer than ref aligns without the 0.43 ref px half-pixel shift."""
    rng = np.random.default_rng(5)
    ref = cv2.GaussianBlur(rng.uniform(0, 255, (400, 400)).astype(np.float32), (0, 0), 3)
    ref = cv2.normalize(ref, None, 10, 250, cv2.NORM_MINMAX)
    # mon covers ref[100:200, 120:220] at 7x the resolution, sampled at mon pixel centers
    my, mx = np.mgrid[0:700, 0:700].astype(np.float32)
    ref_x, ref_y = 120 + (mx + 0.5) / 7 - 0.5, 100 + (my + 0.5) / 7 - 0.5
    mon = cv2.remap(ref, ref_x, ref_y, cv2.INTER_CUBIC)
    truth = _truth(120, 100, 7)

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, 3, -2))

    # Scaling the working grid without the half-pixel term put the center 0.43 px off
    center = np.array([[350.0, 350.0]])
    found = global_align._footprint_points(alignment.matrix, center)
    error = found - global_align._footprint_points(truth, center)
    assert np.abs(error).max() < 0.05
    # The corners carry ECC's own scale uncertainty on a 100 ref px footprint
    assert _center_error(alignment.matrix, truth, size=700) < 1.5


def test_translation_search_is_kept_when_ecc_does_not_converge(fine_pair, monkeypatch, caplog):
    """Without any ECC run converging, the translation search result beats the raw prior."""
    mon, ref, truth = fine_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)
    monkeypatch.setattr(global_align, "_refine_with_ecc", lambda *args: (None, float("nan")))
    def no_sift(*args, **kwargs):
        raise RuntimeError("Too few good matches")

    monkeypatch.setattr(global_align, "_sift_homography", no_sift)

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, 40, -30))

    assert "keeping the shift estimate unrefined" in caplog.text
    assert _center_error(alignment.matrix, truth) < 1


def test_prior_maps_pixel_centers():
    """Geotransforms give pixel corners; the prior maps OpenCV pixel centers."""

    class Image:
        projection = "x"
        spatial_ref = type("SR", (), {"IsSame": lambda self, other: True})()

        def __init__(self, x_min, y_max, res):
            self.x_min, self.y_max, self.x_res, self.y_res = x_min, y_max, res, -res

    prior = global_align._prior_from_georefs(Image(300.0, 900.0, 5.0), Image(0.0, 1000.0, 10.0))

    # mon pixel (0, 0) covers ref's [30, 30.5] x [10, 10.5] corner square, centered on 30.25
    # in corner coordinates, so 29.75 in OpenCV's center ones
    assert global_align._footprint_points(prior, np.array([[0.0, 0.0]]))[0] == pytest.approx(
        [29.75, 9.75]
    )


def test_output_grid_keeps_the_finer_resolution():
    """A mon 7.2x finer than ref gets a grid 8x finer; a coarser one gets ref's grid."""
    valid = np.ones((720, 720), dtype=bool)
    fine = global_align._output_grid(_truth(100, 50, 7.2), valid, (400, 400))
    coarse = global_align._output_grid(_truth(100, 50, 0.5), np.ones((100, 100), bool), (400, 400))

    assert fine.factor == 8
    # 720 mon px of 1/7.2 ref px from ref pixel (100, 50): 100 ref px, 800 output px
    assert (fine.x0, fine.y0, fine.width, fine.height) == (100, 50, 800, 800)
    assert coarse.factor == 1
    # 100 mon px of 2 ref px from (100, 50), clipped to ref's 400 px
    assert (coarse.x0, coarse.y0, coarse.width, coarse.height) == (100, 50, 200, 200)


def test_output_grid_covers_the_valid_data_only():
    """No-data around a rotated scene is left out of the grid."""
    valid = np.zeros((700, 700), dtype=bool)
    valid[200:400, 300:500] = True

    grid = global_align._output_grid(_truth(100, 50, 7), valid, (400, 400))

    # Data spans mon px [300, 500) x [200, 400): ref px [142.9, 171.4) x [78.6, 107.1).
    # Columns are exact, rows widened by at most the outline's 16 mon px sampling
    # step (2.3 ref px)
    assert grid.x0 == 142 and 76 <= grid.y0 <= 78
    assert 172 <= grid.x0 + grid.width // grid.factor <= 175
    assert 108 <= grid.y0 + grid.height // grid.factor <= 110


def _geotiff(path, array, x_min, y_max, res, nodata=None):
    srs = gdal.osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    global_align._write_geotiff(path, array, x_min, y_max, res, -res, srs.ExportToWkt(), nodata)
    return GdalRasterImage(str(path))


def _on_grid(texture, image, ref_res, x_min=500000.0, y_max=5000000.0):
    """`texture`, a ref at `ref_res` from (x_min, y_max), resampled onto `image`'s grid."""
    factor = round(ref_res / image.x_res)
    col, row = (image.x_min - x_min) / ref_res, (y_max - image.y_max) / ref_res
    offset = np.array([[1.0, 0.0, -col], [0.0, 1.0, -row], [0.0, 0.0, 1.0]])
    to_grid = global_align._pixel_scale(factor, factor) @ offset
    size = (image.x_size, image.y_size)
    return cv2.warpPerspective(texture.astype(np.float32), to_grid, size, flags=cv2.INTER_CUBIC)


def test_aligned_output_keeps_the_monitored_resolution(tmp_path):
    """mon 4x finer than ref comes out at its own 5 m over its footprint, on ref's pixel edges."""
    rng = np.random.default_rng(6)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (600, 600)).astype(np.float32), (0, 0), 3)
    texture = cv2.normalize(texture, None, 100, 4000, cv2.NORM_MINMAX).astype(np.uint16)
    # ref at 20 m, mon at 5 m covering ref pixels [150, 350) x [180, 380), georeferenced 60 m off
    ref = _geotiff(tmp_path / "ref.tif", texture, 500000.0, 5000000.0, 20.0)
    mon_array = cv2.resize(texture[150:350, 180:380], (800, 800), interpolation=cv2.INTER_CUBIC)
    mon_x, mon_y = 500000.0 + 180 * 20 + 60, 5000000.0 - 150 * 20 - 40
    mon = _geotiff(tmp_path / "mon.tif", mon_array, mon_x, mon_y, 5.0)
    out = tmp_path / "out"
    out.mkdir()

    aligned, _, _ = global_align.apply_global_alignment(mon, ref, None, out)

    assert (aligned.x_res, aligned.y_res) == (5.0, -5.0)
    # On ref's pixel edges, within mon's footprint and a pixel of margin
    assert (aligned.x_min - 500000.0) % 20 == 0 and (5000000.0 - aligned.y_max) % 20 == 0
    assert 500000.0 + 179 * 20 <= aligned.x_min <= 500000.0 + 180 * 20
    assert aligned.x_size <= 808 and aligned.y_size <= 808
    # Same content as ref on that grid, 5 m detail kept
    inner = (slice(40, -40), slice(40, -40))
    expected = _on_grid(texture, aligned, 20.0)[inner]
    assert np.corrcoef(aligned.array[inner].ravel(), expected.ravel())[0, 1] > 0.98
    # Only the aligned monitored image is written
    assert [p.name for p in out.iterdir()] == ["mon_global_aligned.tif"]


def test_coarser_monitored_gets_the_reference_pixel_size(tmp_path):
    """A mon coarser than ref is rendered at ref's pixel size, on ref's own grid."""
    rng = np.random.default_rng(7)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (400, 400)).astype(np.float32), (0, 0), 4)
    texture = cv2.normalize(texture, None, 100, 4000, cv2.NORM_MINMAX).astype(np.uint16)
    ref = _geotiff(tmp_path / "ref.tif", texture, 500000.0, 5000000.0, 10.0)
    mon_array = cv2.resize(texture[100:300, 80:280], (100, 100), interpolation=cv2.INTER_AREA)
    mon = _geotiff(tmp_path / "mon.tif", mon_array, 500000.0 + 80 * 10, 5000000.0 - 100 * 10, 20.0)
    out = tmp_path / "out"
    out.mkdir()

    aligned, _, _ = global_align.apply_global_alignment(mon, ref, None, out)

    assert (aligned.x_res, aligned.y_res) == (10.0, -10.0)
    assert (aligned.x_min - 500000.0) % 10 == 0 and (5000000.0 - aligned.y_max) % 10 == 0
    inner = (slice(10, -10), slice(10, -10))
    expected = _on_grid(texture, aligned, 10.0)[inner]
    assert np.corrcoef(aligned.array[inner].ravel(), expected.ravel())[0, 1] > 0.95
