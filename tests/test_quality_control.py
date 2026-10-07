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
"""Tests for the matching confidence of karios process."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr

import karios
from karios.accuracy_analysis import quality_control
from karios.accuracy_analysis.quality_control import (
    DOUBTFUL,
    RELIABLE,
    UNRELIABLE,
    assess_quality,
)
from karios.api.config import RuntimeConfiguration
from karios.api.core import KariosAPI
from karios.core.configuration import ProcessingConfiguration

CONFIG_FILE = Path(karios.__file__).parent / "configuration" / "processing_configuration.json"


def _points(n=400, zncc=0.9, noise=0.05, seed=0):
    """Key points on a grid shifting by (0.5, -0.3) px plus `noise`, all confident."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    x0, y0 = np.meshgrid(np.arange(side) * 10.0, np.arange(side) * 10.0)
    return pd.DataFrame(
        {
            "x0": x0.ravel()[:n],
            "y0": y0.ravel()[:n],
            "dx": 0.5 + rng.normal(0, noise, n),
            "dy": -0.3 + rng.normal(0, noise, n),
            "score": 0.9,
            "zncc_score": zncc,
        }
    )


def test_coherent_correlated_points_are_reliable():
    quality = assess_quality(_points(), 0.4, detected_points=1000)

    assert quality.verdict == RELIABLE
    assert quality.confidence == pytest.approx(1.0)
    assert quality.median_zncc == pytest.approx(0.9)
    assert quality.coherent_fraction == pytest.approx(1.0)
    assert quality.tracking_ratio == pytest.approx(0.4)
    assert quality.reasons == []


def test_random_uncorrelated_points_are_unreliable():
    """Shifts scattered over a few pixels with no patch correlation: unrelated images."""
    points = _points(zncc=0.03, noise=2.0)

    quality = assess_quality(points, 0.4, detected_points=20000)

    assert quality.verdict == UNRELIABLE
    assert quality.confidence < 0.1
    assert quality.coherent_fraction < 0.3
    assert len(quality.reasons) == 4
    assert "unrelated" in quality.reasons[-1]


def test_middling_indicators_are_doubtful():
    """Half the points from a coherent field, half random, as with clouds over half the scene."""
    points = _points(zncc=0.35, noise=0.05)
    points.loc[::2, ["dx", "dy"]] += np.random.default_rng(1).normal(0, 3.0, (200, 2))

    quality = assess_quality(points, 0.4, detected_points=3000)

    assert quality.verdict == DOUBTFUL
    assert 0.3 <= quality.confidence < 0.7


def test_points_below_the_threshold_are_left_out():
    points = _points()
    noisy = _points(zncc=0.0, noise=3.0, seed=1).assign(score=0.2, x0=lambda df: df.x0 + 5)

    quality = assess_quality(pd.concat([points, noisy]), 0.4, detected_points=2000)

    assert quality.confident_points == 400
    assert quality.tracked_points == 800
    assert quality.coherent_fraction == pytest.approx(1.0)


def test_missing_zncc_and_detected_count_use_the_other_indicators():
    """No ZNCC after a large shift, no detected count after --resume."""
    points = _points().drop(columns="zncc_score")

    quality = assess_quality(points, 0.4, detected_points=None)

    assert quality.median_zncc is None and quality.tracking_ratio is None
    assert quality.verdict == RELIABLE

    all_nan = assess_quality(_points(zncc=np.nan), 0.4, detected_points=None)
    assert all_nan.median_zncc is None


def test_too_few_confident_points_are_unreliable():
    points = _points(n=quality_control.MIN_CONFIDENT_POINTS - 1)

    quality = assess_quality(points, 0.4, detected_points=100)

    assert quality.verdict == UNRELIABLE
    assert quality.confidence == 0.0
    assert f"only {len(points)} key points" in quality.reasons[0]


def test_no_point_at_all_is_unreliable():
    empty = pd.DataFrame(columns=["x0", "y0", "dx", "dy", "score"])

    quality = assess_quality(empty, 0.4, detected_points=0)

    assert quality.verdict == UNRELIABLE
    assert quality.tracking_ratio is None


def test_to_dict_has_no_nan():
    quality = assess_quality(_points(), 0.4, detected_points=None)
    quality.median_zncc = float("nan")

    data = quality.to_dict()

    assert data["median_zncc"] is None
    json.dumps(data, allow_nan=False)


def _write_raster(path, array):
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), array.shape[1], array.shape[0], 1, gdal.GDT_UInt16
    )
    dataset.SetGeoTransform((600000.0, 10.0, 0.0, 5000000.0, 0.0, -10.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset = None


def _texture(seed, size=400):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 2.0)
    return ((image - image.min()) / (image.max() - image.min()) * 4000 + 100).astype(np.uint16)


def _api(out_dir):
    runtime_config = RuntimeConfiguration(
        output_directory=out_dir,
        pixel_size=None,
        title_prefix=None,
        gen_kp_mask=False,
        gen_delta_raster=False,
        generate_kp_chips=False,
        dem_description=None,
        enable_large_shift_detection=False,
    )
    return KariosAPI(ProcessingConfiguration.from_file(CONFIG_FILE), runtime_config)


def _analyze(tmp_path, ref, mon):
    _write_raster(tmp_path / "ref.tif", ref)
    _write_raster(tmp_path / "mon.tif", mon)
    api = _api(tmp_path / "out")
    match = api.match_images(tmp_path / "mon.tif", tmp_path / "ref.tif")
    return match, api.analyze_accuracy(match)


def test_related_pair_is_reliable(tmp_path):
    ref = _texture(1)
    mon = np.roll(np.roll(ref, 1, axis=0), 2, axis=1)

    match, accuracy = _analyze(tmp_path, ref, mon)

    quality = accuracy.quality
    assert quality.verdict == RELIABLE, quality
    assert match.detected_points > len(match.points) > 0
    assert quality.detected_points == match.detected_points
    summary = json.loads(Path(accuracy.summary_file).read_text(encoding="utf-8"))
    assert summary["quality"]["verdict"] == RELIABLE
    assert summary["statistics"]["ce90"] == pytest.approx(accuracy.ce90)
    assert summary["matched_points"] == len(match.points)


def test_unrelated_pair_is_unreliable(tmp_path):
    _, accuracy = _analyze(tmp_path, _texture(1), _texture(2))

    assert accuracy.quality.verdict == UNRELIABLE, accuracy.quality
    summary = json.loads(Path(accuracy.summary_file).read_text(encoding="utf-8"))
    assert summary["quality"]["verdict"] == UNRELIABLE


def test_no_confident_point_does_not_crash(tmp_path, monkeypatch):
    """Without any key point above the threshold, the statistics are undefined, not an error."""
    _write_raster(tmp_path / "ref.tif", _texture(1, size=100))
    _write_raster(tmp_path / "mon.tif", _texture(1, size=100))
    api = _api(tmp_path / "out")
    match = api.match_images(tmp_path / "mon.tif", tmp_path / "ref.tif")
    match.points["score"] = 0.0

    accuracy = api.analyze_accuracy(match)

    assert accuracy.quality.verdict == UNRELIABLE
    assert math.isnan(accuracy.ce90) and math.isnan(accuracy.mean_x)
    summary = json.loads(Path(accuracy.summary_file).read_text(encoding="utf-8"))
    assert summary["statistics"]["ce90"] is None
