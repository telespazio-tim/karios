#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the SIFT keypoint limit of the global alignment."""

import logging
import re

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from karios.cli import commands
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
