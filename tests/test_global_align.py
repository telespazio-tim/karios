#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the SIFT keypoint limit of the global alignment."""

import logging
import re

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from osgeo import gdal, osr

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


def test_sift_nfeatures_defaults_to_10000(shifted_pair, caplog):
    mon, ref = shifted_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)

    detect_global_alignment(mon, ref)

    assert global_align.SIFT_NFEATURES == 10000
    # This small pair has fewer: all are kept
    assert 300 < min(_keypoint_counts(caplog)) <= max(_keypoint_counts(caplog)) <= 10010


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

    assert calls == [
        {"sift_nfeatures": 5000, "transform_path": tmp_path / "out" / "mon_global_alignment.json"}
    ]
    # The fake alignment raises: align reports the failure in its exit status
    assert result.exit_code == 1, result.output


def test_cli_sift_nfeatures_defaults_to_10000(tmp_path, monkeypatch):
    _, calls = _run_align(tmp_path, monkeypatch)

    assert [call["sift_nfeatures"] for call in calls] == [10000]


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
    frame = global_align._work_frame(mon, ref, truth)
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
    assert np.abs(error).max() < 0.15
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
            self.geo_transform = (x_min, res, 0.0, y_max, 0.0, -res)

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
    factor = ref_res / image.x_res
    col, row = (image.x_min - x_min) / ref_res, (y_max - image.y_max) / ref_res
    offset = np.array([[1.0, 0.0, -col], [0.0, 1.0, -row], [0.0, 0.0, 1.0]])
    to_grid = global_align._pixel_scale(factor, factor) @ offset
    size = (image.x_size, image.y_size)
    return cv2.warpPerspective(texture.astype(np.float32), to_grid, size, flags=cv2.INTER_CUBIC)


def test_aligned_output_keeps_the_monitored_grid(tmp_path):
    """mon 4x finer than ref comes out on its own 5 m grid, moved to where ref sees it."""
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

    assert aligned.spatial_ref.IsSame(mon.spatial_ref)
    assert (aligned.x_res, aligned.y_res) == (5.0, -5.0)
    # On mon's pixel edges, at its true place within a pixel of margin
    assert (aligned.x_min - mon_x) % 5 == 0 and (mon_y - aligned.y_max) % 5 == 0
    assert 503600.0 - 5 <= aligned.x_min <= 503600.0
    assert 4997000.0 <= aligned.y_max <= 4997000.0 + 5
    assert aligned.x_size <= 802 and aligned.y_size <= 802
    # Same content as ref on that grid, 5 m detail kept
    inner = (slice(40, -40), slice(40, -40))
    expected = _on_grid(texture, aligned, 20.0)[inner]
    assert np.corrcoef(aligned.array[inner].ravel(), expected.ravel())[0, 1] > 0.98
    # Only the aligned monitored image is written
    assert [p.name for p in out.iterdir()] == ["mon_global_aligned.tif"]


def test_coarser_monitored_keeps_its_pixel_size(tmp_path):
    """A mon coarser than ref keeps its own 20 m pixels, on its own grid."""
    rng = np.random.default_rng(7)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (400, 400)).astype(np.float32), (0, 0), 4)
    texture = cv2.normalize(texture, None, 100, 4000, cv2.NORM_MINMAX).astype(np.uint16)
    ref = _geotiff(tmp_path / "ref.tif", texture, 500000.0, 5000000.0, 10.0)
    mon_array = cv2.resize(texture[100:300, 80:280], (100, 100), interpolation=cv2.INTER_AREA)
    mon = _geotiff(tmp_path / "mon.tif", mon_array, 500000.0 + 80 * 10, 5000000.0 - 100 * 10, 20.0)
    out = tmp_path / "out"
    out.mkdir()

    aligned, _, _ = global_align.apply_global_alignment(mon, ref, None, out)

    assert aligned.spatial_ref.IsSame(mon.spatial_ref)
    assert (aligned.x_res, aligned.y_res) == (20.0, -20.0)
    assert (aligned.x_min - mon.x_min) % 20 == 0 and (mon.y_max - aligned.y_max) % 20 == 0
    inner = (slice(5, -5), slice(5, -5))
    expected = _on_grid(cv2.GaussianBlur(texture.astype(np.float32), (0, 0), 1), aligned, 10.0)
    assert np.corrcoef(aligned.array[inner].ravel(), expected[inner].ravel())[0, 1] > 0.95


def test_whole_pixel_correction_moves_the_georeferencing_only(tmp_path, monkeypatch):
    """Ref seeing mon 2 ref px east and 1 north of its georeferencing: same pixels, moved grid."""
    rng = np.random.default_rng(8)
    mon_array = rng.integers(1, 4000, (120, 100), dtype=np.uint16)
    mon = _geotiff(tmp_path / "mon.tif", mon_array, 500000.0, 5000000.0, 10.0)
    ref = _geotiff(tmp_path / "ref.tif", np.ones((200, 200), np.uint16), 499000.0, 5001000.0, 20.0)
    georefs = global_align._prior_from_georefs(mon, ref)
    shift = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    alignment = global_align.GlobalAlignment(matrix=shift @ georefs, n_inliers=4, n_matches=4)
    monkeypatch.setattr(global_align, "detect_global_alignment", lambda *args, **kwargs: alignment)
    out = tmp_path / "out"
    out.mkdir()

    aligned, _, _ = global_align.apply_global_alignment(mon, ref, None, out)

    assert aligned.geo_transform == (500040.0, 10.0, 0.0, 5000020.0, 0.0, -10.0)
    assert np.array_equal(aligned.array, mon_array)


def test_monitored_without_crs_is_written_in_the_reference_crs(tmp_path, monkeypatch):
    """No CRS to keep: the output is nested in ref's grid, in ref's CRS, as before."""
    rng = np.random.default_rng(9)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (300, 300)).astype(np.float32), (0, 0), 3)
    texture = cv2.normalize(texture, None, 100, 4000, cv2.NORM_MINMAX).astype(np.uint16)
    ref = _geotiff(tmp_path / "ref.tif", texture, 500000.0, 5000000.0, 20.0)
    path = tmp_path / "mon.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), 400, 400, 1, gdal.GDT_UInt16)
    dataset.GetRasterBand(1).WriteArray(
        cv2.resize(texture[50:150, 60:160], (400, 400), interpolation=cv2.INTER_CUBIC)
    )
    dataset = None
    mon = GdalRasterImage(str(path))
    alignment = global_align.GlobalAlignment(matrix=_truth(60, 50, 4), n_inliers=4, n_matches=4)
    monkeypatch.setattr(global_align, "detect_global_alignment", lambda *args, **kwargs: alignment)
    out = tmp_path / "out"
    out.mkdir()

    aligned, _, _ = global_align.apply_global_alignment(mon, ref, None, out)

    assert aligned.spatial_ref.IsSame(ref.spatial_ref)
    assert aligned.geo_transform == (501200.0, 5.0, 0.0, 4999000.0, 0.0, -5.0)


def test_flann_matching_recovers_the_shift(shifted_pair, monkeypatch, caplog):
    """Above BRUTE_FORCE_MAX_PAIRS the KD-tree matching finds the same alignment."""
    mon, ref = shifted_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)
    monkeypatch.setattr(global_align, "BRUTE_FORCE_MAX_PAIRS", 0)

    alignment = detect_global_alignment(mon, ref)

    assert "matched by FLANN KD-tree" in caplog.text
    assert alignment.matrix[0, 2] == pytest.approx(5, abs=0.5)
    assert alignment.matrix[1, 2] == pytest.approx(3, abs=0.5)


def test_flann_matching_finds_the_true_neighbours(monkeypatch):
    """Each query is a slightly noisy copy of one train descriptor: both matchers find it."""
    rng = np.random.default_rng(8)
    train = rng.uniform(0, 100, (5000, 128)).astype(np.float32)
    picks = rng.choice(len(train), 500, replace=False)
    query = train[picks] + rng.normal(0, 1, (500, 128)).astype(np.float32)

    brute = [pair[0].trainIdx for pair in global_align._knn_match(query, train, k=2)]
    monkeypatch.setattr(global_align, "BRUTE_FORCE_MAX_PAIRS", 0)
    flann = [pair[0].trainIdx for pair in global_align._knn_match(query, train, k=2)]

    assert brute == picks.tolist()
    # Approximate search: the KD-tree may miss a few
    assert np.mean(np.array(flann) == picks) > 0.95


def test_matching_takes_more_descriptors_than_brute_force_can(monkeypatch):
    """OpenCV's brute force refuses 262144 train descriptors; the KD-tree takes them."""
    rng = np.random.default_rng(9)
    train = rng.uniform(0, 100, (262_144, 32)).astype(np.float32)
    query = train[:200] + rng.normal(0, 0.5, (200, 32)).astype(np.float32)
    monkeypatch.setattr(global_align, "BRUTE_FORCE_MAX_PAIRS", 0)

    matches = global_align._knn_match(query, train, k=2)

    found = [pair[0].trainIdx for pair in matches]
    assert np.mean(np.array(found) == np.arange(200)) > 0.95


@pytest.mark.parametrize("offset", [(2.0, -2.5), (-1.5, 3.0), (0.7, 0.4)])
def test_ecc_corrects_the_starting_error(offset):
    """ECC from a start a few pixels off lands on the truth, not twice as far on the other side."""
    rng = np.random.default_rng(1)
    ref = cv2.GaussianBlur(rng.uniform(0, 255, (1000, 1000)).astype(np.float32), (0, 0), 3)
    ref = cv2.normalize(ref, None, 10, 250, cv2.NORM_MINMAX).astype(np.uint8)
    mon = ref[300:700, 200:600].copy()
    truth = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 300.0], [0.0, 0.0, 1.0]])

    refined, score = global_align._refine_with_ecc(mon, ref, _offset(truth, *offset))

    assert score > 0.99
    assert _center_error(refined, truth, size=400) < 0.1


@pytest.fixture(name="large_texture", scope="module")
def large_texture_fixture():
    rng = np.random.default_rng(10)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (700, 900)).astype(np.float32), (0, 0), 2)
    return cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _sift():
    return cv2.SIFT_create(
        nfeatures=0,
        contrastThreshold=global_align.SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=global_align.SIFT_EDGE_THRESHOLD,
    )


def test_tiled_sift_finds_the_whole_image_keypoints(large_texture, monkeypatch):
    """Tiles with a margin find the keypoints SIFT finds on the whole image."""
    whole, whole_desc = _sift().detectAndCompute(large_texture, None)
    monkeypatch.setattr(global_align, "SIFT_TILE_PX", 256)
    monkeypatch.setattr(global_align, "SIFT_TILE_MARGIN_PX", 64)

    tiled, tiled_desc = global_align._detect_sift(_sift(), large_texture, 0)

    # A few keypoints larger than the margin may differ next to tile edges
    assert len(tiled) == pytest.approx(len(whole), rel=0.02)
    whole_pts = np.array([kp.pt for kp in whole])
    distances = [np.hypot(*(whole_pts - kp.pt).T).min() for kp in tiled]
    assert np.mean(np.array(distances) < 0.01) > 0.97
    assert tiled_desc.shape == (len(tiled), 128)


def test_tiled_sift_keeps_the_strongest_keypoints(large_texture, monkeypatch):
    monkeypatch.setattr(global_align, "SIFT_TILE_PX", 256)
    every, _ = global_align._detect_sift(_sift(), large_texture, 0)

    strongest, descriptors = global_align._detect_sift(_sift(), large_texture, 100)

    assert len(strongest) == 100 and len(descriptors) == 100
    threshold = sorted((kp.response for kp in every), reverse=True)[99]
    assert min(kp.response for kp in strongest) >= threshold


def test_tiled_sift_aligns_like_the_whole_image(shifted_pair, monkeypatch):
    """Tiling SIFT does not change the alignment."""
    mon, ref = shifted_pair
    monkeypatch.setattr(global_align, "SIFT_TILE_PX", 200)

    alignment = detect_global_alignment(mon, ref)

    assert alignment.matrix[0, 2] == pytest.approx(5, abs=0.5)
    assert alignment.matrix[1, 2] == pytest.approx(3, abs=0.5)


def _rotated_wgs84(lon, lat, step, angle_deg, mirror):
    """Geotransform in degrees, `step` degrees per pixel, rotated and optionally mirrored."""
    a = np.radians(angle_deg)
    col = np.array([np.cos(a), np.sin(a)]) * step * (-1 if mirror else 1)
    row = np.array([np.sin(a), -np.cos(a)]) * step
    return (lon, col[0], row[0], lat, col[1], row[1])


@pytest.fixture(name="wgs84_pair", scope="module")
def wgs84_pair_fixture(tmp_path_factory):
    """ref in UTM 31N at 20 m; mon in WGS 84 at ~5 m, rotated 30° and mirrored.

    mon's pixels are sampled from ref at their true place; its georeferencing
    then claims a place 200 m east and 150 m north of it. Returns (mon, ref,
    truth), truth mapping mon pixel centers to ref's.
    """
    tmp = tmp_path_factory.mktemp("wgs84")
    rng = np.random.default_rng(11)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (700, 700)).astype(np.float32), (0, 0), 3)
    texture = cv2.normalize(texture, None, 100, 4000, cv2.NORM_MINMAX).astype(np.uint16)
    ref_geo = (600000.0, 20.0, 0.0, 4830000.0, 0.0, -20.0)
    ref = _geotiff(tmp / "ref.tif", texture, 600000.0, 4830000.0, 20.0)

    wgs84, utm = osr.SpatialReference(), osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    utm.ImportFromEPSG(32631)
    for srs in (wgs84, utm):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_utm = osr.CoordinateTransformation(wgs84, utm)
    to_wgs84 = osr.CoordinateTransformation(utm, wgs84)
    # True grid: from UTM (604000, 4826000), 4.5e-5 degrees per pixel (~4-5 m)
    lon0, lat0, _ = to_wgs84.TransformPoint(604000.0, 4826000.0)
    true_geo = _rotated_wgs84(lon0, lat0, 4.5e-5, 30.0, mirror=True)

    size = 500
    rows, cols = np.mgrid[0:size, 0:size].astype(np.float64)
    lon = true_geo[0] + (cols + 0.5) * true_geo[1] + (rows + 0.5) * true_geo[2]
    lat = true_geo[3] + (cols + 0.5) * true_geo[4] + (rows + 0.5) * true_geo[5]
    east_north = np.array(to_utm.TransformPoints(np.column_stack([lon.ravel(), lat.ravel()])))
    ref_x = (east_north[:, 0] - ref_geo[0]) / ref_geo[1] - 0.5
    ref_y = (east_north[:, 1] - ref_geo[3]) / ref_geo[5] - 0.5
    map_x = ref_x.reshape(size, size).astype(np.float32)
    map_y = ref_y.reshape(size, size).astype(np.float32)
    mon_array = cv2.remap(texture.astype(np.float32), map_x, map_y, cv2.INTER_CUBIC)
    mon_array = np.clip(mon_array, 1, 65535).astype(np.uint16)

    # Georeferencing off by 200 m east and 150 m north
    off_lon, off_lat, _ = to_wgs84.TransformPoint(604200.0, 4826150.0)
    claimed = list(true_geo)
    claimed[0] += off_lon - lon0
    claimed[3] += off_lat - lat0
    path = tmp / "mon.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), size, size, 1, gdal.GDT_UInt16)
    dataset.SetProjection(wgs84.ExportToWkt())
    dataset.SetGeoTransform(claimed)
    dataset.GetRasterBand(1).WriteArray(mon_array)
    dataset = None

    # A homography fits the true mapping within 0.01 ref px over this 2.5 km footprint
    grid = (slice(None, None, 50), slice(None, None, 50))
    sample = np.column_stack([cols[grid].ravel(), rows[grid].ravel()])
    true_ref = np.column_stack([map_x[grid].ravel(), map_y[grid].ravel()]).astype(np.float64)
    truth, _ = cv2.findHomography(sample, true_ref, 0)
    return GdalRasterImage(str(path)), ref, truth


def test_prior_is_reprojected_across_crs(wgs84_pair):
    """A rotated, mirrored WGS 84 mon gets a prior on a UTM ref, off by its georeferencing only."""
    mon, ref, truth = wgs84_pair

    prior = global_align._prior_from_georefs(mon, ref)

    corners = np.array([[0.0, 0.0], [500.0, 0.0], [500.0, 500.0], [0.0, 500.0]])
    found = global_align._footprint_points(prior, corners)
    offset = found - global_align._footprint_points(truth, corners)
    # 200 m east, 150 m north: +10 ref px in x, -7.5 in y (ref rows go south)
    assert offset[:, 0] == pytest.approx(10, abs=0.2)
    assert offset[:, 1] == pytest.approx(-7.5, abs=0.2)
    # Mirrored grid: the prior's linear part reverses orientation
    assert np.linalg.det(prior[:2, :2]) < 0


def test_rotated_mirrored_wgs84_mon_aligns_on_utm_ref(wgs84_pair):
    mon, ref, truth = wgs84_pair

    alignment = detect_global_alignment(
        mon.array, ref.array, prior=global_align._prior_from_georefs(mon, ref)
    )

    # From 12.5 ref px off: the center lands on the truth, the corners carry
    # ECC's scale uncertainty on a 125 ref px footprint
    center = np.array([[250.0, 250.0]])
    found = global_align._footprint_points(alignment.matrix, center)
    assert np.abs(found - global_align._footprint_points(truth, center)).max() < 0.05
    assert _center_error(alignment.matrix, truth, size=500) < 0.6


def test_rotated_mirrored_wgs84_mon_stays_in_wgs84(wgs84_pair, tmp_path, monkeypatch):
    """The aligned output keeps mon's WGS 84, north-up at its pixel area, and overlays ref.

    Resampled by nearest neighbour, it holds mon's own pixel values only.
    """
    mon, ref, truth = wgs84_pair
    alignment = global_align.GlobalAlignment(matrix=truth, n_inliers=4, n_matches=4)
    monkeypatch.setattr(global_align, "detect_global_alignment", lambda *args, **kwargs: alignment)

    aligned, _, _ = global_align.apply_global_alignment(mon, ref, None, tmp_path)

    assert aligned.spatial_ref.IsSame(mon.spatial_ref)
    assert aligned.geo_transform[2] == aligned.geo_transform[4] == 0
    assert (aligned.x_res, aligned.y_res) == pytest.approx((4.5e-5, -4.5e-5))
    assert np.isin(aligned.array, np.append(mon.array, 0)).all()
    # ref resampled onto the output grid matches it best there, not a quarter pixel aside:
    # on average, nearest neighbour moving each pixel by up to half a pixel
    valid = cv2.erode((aligned.array > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0

    def error(dx, dy):
        x, y = dx * aligned.x_res, dy * aligned.y_res
        bounds = (aligned.x_min + x, aligned.y_min + y, aligned.x_max + x, aligned.y_max + y)
        on_grid = gdal.Warp(
            "",
            ref.filepath,
            format="MEM",
            dstSRS=aligned.projection,
            outputBounds=bounds,
            width=aligned.x_size,
            height=aligned.y_size,
            resampleAlg="cubic",
        ).ReadAsArray()
        return np.mean(np.abs(aligned.array.astype(float) - on_grid)[valid])

    aside = [error(dx, dy) for dx, dy in [(0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25)]]
    assert error(0, 0) < min(aside)


def test_cli_quotes_the_reference_in_the_printed_gdalwarp(tmp_path, monkeypatch):
    """A reference named like a shell substitution is quoted in the command to paste."""
    grid = {"x_size": 10, "y_size": 10, "x_res": 5.0, "y_res": -5.0, "x_min": 0.0, "y_max": 50.0}
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    aligned = type(
        "Aligned", (), {**grid, "file_name": "mon_global_aligned.tif", "spatial_ref": srs}
    )()
    reference = type("Ref", (), {"x_res": 10.0, "y_res": -10.0})()
    alignment = global_align.GlobalAlignment(matrix=np.eye(3), n_inliers=4, n_matches=4)
    monkeypatch.setattr(commands, "GdalRasterImage", lambda path: reference)
    monkeypatch.setattr(
        commands, "apply_global_alignment", lambda *args, **kwargs: (aligned, None, alignment)
    )
    mon, ref = tmp_path / "mon.tif", tmp_path / "$(touch pwned).tif"
    mon.touch()
    ref.touch()

    result = CliRunner().invoke(
        commands.cli, ["align", str(mon), str(ref), "--out", str(tmp_path / "out"), "--no-log-file"]
    )

    assert result.exit_code == 0, result.output
    command = next(line for line in result.output.splitlines() if "gdalwarp" in line)
    assert f"'{ref}'" in command
    # The reference is resampled into the output's CRS, at its exact pixel size
    assert "-t_srs EPSG:32631 " in command
    assert "-tr 5.0 5.0 " in command


def test_raster_windows_read_only_the_requested_window(tmp_path):
    texture = np.arange(60 * 80, dtype=np.uint16).reshape(60, 80)
    image = _geotiff(tmp_path / "ref.tif", texture, 500000.0, 5000000.0, 10.0)
    windows = global_align.RasterWindows(image)

    assert windows.shape == (60, 80)
    assert np.array_equal(windows[10:25, 5:40], texture[10:25, 5:40])
    assert np.array_equal(windows[50:999, 70:999], texture[50:, 70:])  # clipped like a slice
    assert image._array is None  # windows never loaded the whole band
    assert np.array_equal(np.asarray(windows), texture)


def test_alignment_with_a_prior_reads_the_search_window_only(tmp_path):
    rng = np.random.default_rng(6)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (600, 600)).astype(np.float32), (0, 0), 3)
    texture = cv2.normalize(texture, None, 100, 4000, cv2.NORM_MINMAX).astype(np.uint16)
    ref = _geotiff(tmp_path / "ref.tif", texture, 500000.0, 5000000.0, 20.0)
    mon_array = cv2.resize(texture[150:350, 180:380], (800, 800), interpolation=cv2.INTER_CUBIC)
    mon = _geotiff(tmp_path / "mon.tif", mon_array, 500000.0 + 180 * 20, 5000000.0 - 150 * 20, 5.0)
    out = tmp_path / "out"
    out.mkdir()

    global_align.apply_global_alignment(mon, ref, None, out)

    assert ref._array is None


@pytest.fixture(name="small_footprint_pair", scope="module")
def small_footprint_pair_fixture():
    """mon 4x finer than ref, a 100 ref px footprint at (500, 450) in a 1000 px ref."""
    rng = np.random.default_rng(12)
    ref = cv2.GaussianBlur(rng.uniform(0, 255, (1000, 1000)).astype(np.float32), (0, 0), 3)
    ref = cv2.normalize(ref, None, 10, 250, cv2.NORM_MINMAX)
    mon = cv2.resize(ref[450:550, 500:600], (400, 400), interpolation=cv2.INTER_CUBIC)
    return mon, ref, _truth(500, 450, 4)


def test_georeferencing_off_by_more_than_the_window_widens_it(small_footprint_pair, caplog):
    """70 ref px off with a 50 px margin: the window doubles and the alignment is found."""
    mon, ref, truth = small_footprint_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, 70, -65))

    assert "widening the search margin" in caplog.text
    assert _center_error(alignment.matrix, truth, size=400) < 0.5


def test_georeferencing_within_the_window_does_not_widen_it(small_footprint_pair, caplog):
    mon, ref, truth = small_footprint_pair
    caplog.set_level(logging.INFO, logger=global_align.__name__)

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, 15, -10))

    assert "widening" not in caplog.text
    assert _center_error(alignment.matrix, truth, size=400) < 0.5


def test_only_the_best_probe_is_refined_to_convergence(fine_pair, monkeypatch):
    """Every start gets a short ECC probe; the full ECC runs once, from the winner."""
    mon, ref, truth = fine_pair
    calls = []
    refine = global_align._refine_with_ecc

    def spy(mon_u8, ref_u8, init, max_iters=global_align.ECC_MAX_ITERS):
        calls.append(max_iters)
        return refine(mon_u8, ref_u8, init, max_iters)

    monkeypatch.setattr(global_align, "_refine_with_ecc", spy)

    alignment = detect_global_alignment(mon, ref, prior=_offset(truth, 40, -30))

    assert calls.count(global_align.ECC_MAX_ITERS) == 1
    assert calls.count(global_align.ECC_PROBE_ITERS) == len(calls) - 1 >= 2
    assert _center_error(alignment.matrix, truth) < 0.5


def test_capping_tiles_keeps_the_same_strongest_keypoints(large_texture, monkeypatch):
    """Capping each tile at N gives the image's N strongest, in the same order."""
    monkeypatch.setattr(global_align, "SIFT_TILE_PX", 256)

    def sift(nfeatures):
        return cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=global_align.SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=global_align.SIFT_EDGE_THRESHOLD,
        )

    uncapped, uncapped_desc = global_align._detect_sift(sift(0), large_texture, 150)
    capped, capped_desc = global_align._detect_sift(sift(150), large_texture, 150)

    def key(keypoints):
        return [(kp.pt, kp.size, kp.angle, kp.response) for kp in keypoints]

    assert len(capped) == 150
    assert key(capped) == key(uncapped)
    assert np.array_equal(capped_desc, uncapped_desc)


def test_aligned_integer_pixels_are_rounded_not_truncated():
    """Interpolated values go back to integers by rounding: no -0.5 DN bias."""
    values = np.array([[0.4, 0.6, 1.5, 2.49], [65535.7, -3.2, 100.0, 7.51]], dtype=np.float32)

    as_uint16 = global_align._to_dtype(values, np.uint16)

    assert as_uint16.tolist() == [[0, 1, 2, 2], [65535, 0, 100, 8]]
    assert global_align._to_dtype(values, np.float32) is not values
    assert np.array_equal(global_align._to_dtype(values, np.float32), values)


def test_aligned_output_has_no_rounding_bias(tmp_path, monkeypatch):
    """A smooth integer image shifted by half a pixel keeps its mean; truncation lowered it."""
    yy, xx = np.mgrid[0:300, 0:300]
    texture = (2000 + 900 * np.sin(xx / 13.0) * np.cos(yy / 17.0)).astype(np.uint16)
    ref = _geotiff(tmp_path / "ref.tif", texture, 500000.0, 5000000.0, 10.0)
    mon = _geotiff(tmp_path / "mon.tif", texture, 500000.0, 5000000.0, 10.0)
    out = tmp_path / "out"
    out.mkdir()
    shift = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    alignment = global_align.GlobalAlignment(matrix=shift, n_inliers=4, n_matches=4)
    monkeypatch.setattr(global_align, "detect_global_alignment", lambda *args, **kwargs: alignment)

    aligned, _, _ = global_align.apply_global_alignment(mon, ref, None, out)

    inner = (slice(20, -20), slice(20, -20))
    exact = cv2.warpPerspective(
        texture.astype(np.float32), shift, (300, 300), flags=cv2.INTER_LINEAR
    )
    # On mon's grid from its origin, one column and row more for the shifted edge
    assert (aligned.x_min, aligned.y_max) == (mon.x_min, mon.y_max)
    assert abs(aligned.array[:300, :300][inner].mean() - exact[inner].mean()) < 0.05


def _band_pair(tmp_path):
    """(mon, ref, texture): 5 m mon of 20 m ref pixels [150, 350) x [180, 380), 60 m off."""
    rng = np.random.default_rng(6)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (600, 600)).astype(np.float32), (0, 0), 3)
    texture = cv2.normalize(texture, None, 100, 4000, cv2.NORM_MINMAX).astype(np.uint16)
    ref = _geotiff(tmp_path / "ref.tif", texture, 500000.0, 5000000.0, 20.0)
    mon_array = cv2.resize(texture[150:350, 180:380], (800, 800), interpolation=cv2.INTER_CUBIC)
    mon = _geotiff(tmp_path / "mon.tif", mon_array, 503660.0, 4997040.0, 5.0)
    return mon, ref, texture


def test_saved_transform_round_trips(tmp_path):
    """The saved alignment reads back to the last bit."""
    mon, ref, _ = _band_pair(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    path = tmp_path / "mon_global_alignment.json"

    aligned, _, alignment = global_align.apply_global_alignment(
        mon, ref, None, out, transform_path=path
    )
    transform = global_align.AlignmentTransform.load(path)

    assert np.array_equal(transform.matrix, alignment.matrix)
    assert np.array_equal(transform.prior, global_align._prior_from_georefs(mon, ref))
    assert transform.monitored == global_align.RasterGrid.of(mon)
    assert transform.reference == global_align.RasterGrid.of(ref)
    output = transform.output
    assert (output.x_min, output.x_res, 0.0, output.y_max, 0.0, output.y_res) == tuple(
        aligned.geo_transform
    )
    assert (output.width, output.height) == (aligned.x_size, aligned.y_size)
    # The transform is written next to the aligned image, not by default
    assert sorted(p.name for p in out.iterdir()) == ["mon_global_aligned.tif"]


def test_band_on_the_monitored_grid_gets_the_same_output_grid(tmp_path):
    """Another band of mon's grid is warped exactly like mon, onto the same grid."""
    mon, ref, _ = _band_pair(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    path = tmp_path / "transform.json"
    aligned_mon, _, _ = global_align.apply_global_alignment(
        mon, ref, None, out, transform_path=path
    )
    # A band with other data, and a no-data border mon does not have
    band_array = (mon.array // 2).astype(np.uint16)
    band_array[:, :40] = 0
    band = _geotiff(tmp_path / "band.tif", band_array, mon.x_min, mon.y_max, 5.0)

    aligned_band = global_align.apply_alignment_transform(
        band, global_align.AlignmentTransform.load(path), out
    )

    assert aligned_band.file_name == "band_global_aligned.tif"
    assert aligned_band.geo_transform == aligned_mon.geo_transform
    assert (aligned_band.x_size, aligned_band.y_size) == (aligned_mon.x_size, aligned_mon.y_size)
    assert aligned_band.spatial_ref.IsSame(aligned_mon.spatial_ref)
    # Same warp: half of mon's values, to the rounding of each, where the band has data
    inner = (slice(10, -10), slice(60, -10))
    half = aligned_mon.array[inner].astype(float) / 2
    assert np.abs(aligned_band.array[inner] - half).max() <= 1


def test_band_on_a_coarser_grid_keeps_its_grid(tmp_path):
    """A 20 m band of the 5 m mon is corrected like mon, on its own 20 m grid."""
    mon, ref, texture = _band_pair(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    path = tmp_path / "transform.json"
    global_align.apply_global_alignment(mon, ref, None, out, transform_path=path)
    # Same footprint and georeferencing error as mon, at 20 m
    band = _geotiff(tmp_path / "b20.tif", texture[150:350, 180:380], mon.x_min, mon.y_max, 20.0)

    aligned = global_align.apply_alignment_transform(
        band, global_align.AlignmentTransform.load(path), out
    )

    assert aligned.spatial_ref.IsSame(band.spatial_ref)
    assert (aligned.x_res, aligned.y_res) == (20.0, -20.0)
    # On the band's pixel edges, moved back 60 m west and 40 m south to its true place
    assert (aligned.x_min - band.x_min) % 20 == 0 and (band.y_max - aligned.y_max) % 20 == 0
    assert 503600.0 - 20 <= aligned.x_min <= 503600.0
    assert 4997000.0 <= aligned.y_max <= 4997000.0 + 20
    inner = (slice(5, -5), slice(5, -5))
    expected = _on_grid(texture, aligned, 20.0)[inner]
    assert np.corrcoef(aligned.array[inner].ravel(), expected.ravel())[0, 1] > 0.98


def test_transform_without_georeferencing_rejects_another_grid(tmp_path):
    """Without georeferencing, only images on the monitored grid can be placed."""
    grid = global_align.RasterGrid((0.0, 1.0, 0.0, 0.0, 0.0, 1.0), 100, 100, "")
    output = global_align.OutputFrame("", 0.0, 0.0, 1.0, -1.0, 100, 100, np.eye(3))
    transform = global_align.AlignmentTransform(grid, grid, np.eye(3), None, output)
    path = tmp_path / "other.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), 50, 50, 1, gdal.GDT_UInt16)
    dataset = None

    with pytest.raises(ValueError, match="monitored image's grid"):
        global_align.apply_alignment_transform(GdalRasterImage(str(path)), transform, tmp_path)


def test_loading_another_json_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"klt_matching": {}}')

    with pytest.raises(ValueError, match="not an alignment transform"):
        global_align.AlignmentTransform.load(path)


def _cli_band_pair(tmp_path, monkeypatch):
    """(b04, ref, b03, b04 pixels): b04 seen by ref 2 ref px east and 1 north, b03 on its grid."""
    rng = np.random.default_rng(8)
    mon_array = rng.integers(1, 4000, (120, 100), dtype=np.uint16)
    mon = _geotiff(tmp_path / "b04.tif", mon_array, 500000.0, 5000000.0, 10.0)
    ref = _geotiff(tmp_path / "ref.tif", np.ones((200, 200), np.uint16), 499000.0, 5001000.0, 20.0)
    band = _geotiff(tmp_path / "b03.tif", mon_array // 3, 500000.0, 5000000.0, 10.0)
    georefs = global_align._prior_from_georefs(mon, ref)
    shift = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    alignment = global_align.GlobalAlignment(matrix=shift @ georefs, n_inliers=4, n_matches=4)
    monkeypatch.setattr(global_align, "detect_global_alignment", lambda *args, **kwargs: alignment)
    return mon, ref, band, mon_array


def _invoke(*args):
    return CliRunner().invoke(commands.cli, ["align", *map(str, args), "--no-log-file"])


def test_cli_applies_the_alignment_to_other_bands(tmp_path, monkeypatch):
    """--apply-to warps another band by the alignment estimated on the monitored one."""
    mon, ref, band, mon_array = _cli_band_pair(tmp_path, monkeypatch)
    out = tmp_path / "out"

    result = _invoke(mon.filepath, ref.filepath, "--apply-to", band.filepath, "--out", out)

    assert result.exit_code == 0, result.output
    assert f"--load-transform {out / 'b04_global_alignment.json'}" in result.output
    aligned_mon = GdalRasterImage(str(out / "b04_global_aligned.tif"))
    aligned_band = GdalRasterImage(str(out / "b03_global_aligned.tif"))
    assert aligned_mon.geo_transform == (500040.0, 10.0, 0.0, 5000020.0, 0.0, -10.0)
    assert aligned_band.geo_transform == aligned_mon.geo_transform
    assert np.array_equal(aligned_band.array, mon_array // 3)


def test_cli_loads_a_saved_alignment(tmp_path, monkeypatch):
    """--load-transform warps a band by a saved alignment, without a reference."""
    mon, ref, band, mon_array = _cli_band_pair(tmp_path, monkeypatch)
    first, second = tmp_path / "first", tmp_path / "second"
    assert _invoke(mon.filepath, ref.filepath, "--out", first).exit_code == 0
    monkeypatch.setattr(global_align, "detect_global_alignment", None)  # never estimated again

    result = _invoke(
        band.filepath, "--load-transform", first / "b04_global_alignment.json", "--out", second
    )

    assert result.exit_code == 0, result.output
    aligned = GdalRasterImage(str(second / "b03_global_aligned.tif"))
    assert aligned.geo_transform == (500040.0, 10.0, 0.0, 5000020.0, 0.0, -10.0)
    assert np.array_equal(aligned.array, mon_array // 3)
    # Nothing estimated, so nothing saved
    assert sorted(p.name for p in second.iterdir()) == ["b03_global_aligned.tif"]


@pytest.mark.parametrize("with_reference, with_transform", [(True, True), (False, False)])
def test_cli_needs_a_reference_or_a_saved_alignment(tmp_path, with_reference, with_transform):
    """Exactly one of REFERENCE_IMAGE and --load-transform tells how to align."""
    image = _geotiff(tmp_path / "b.tif", np.ones((10, 10), np.uint16), 0.0, 100.0, 10.0)
    transform = tmp_path / "t.json"
    transform.write_text("{}")
    args = [image.filepath] + [image.filepath] * with_reference
    args += ["--load-transform", transform] * with_transform

    result = _invoke(*args)

    assert result.exit_code == 2
    assert "REFERENCE_IMAGE" in result.output and "--load-transform" in result.output


def test_cli_fails_on_a_bad_saved_alignment(tmp_path):
    transform = tmp_path / "bad.json"
    transform.write_text("{}")
    image = _geotiff(tmp_path / "b.tif", np.ones((10, 10), np.uint16), 0.0, 100.0, 10.0)

    result = _invoke(image.filepath, "--load-transform", transform, "--out", tmp_path / "out")

    assert result.exit_code == 1
