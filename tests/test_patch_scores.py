#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The vectorized patch scores reproduce the per-point ZNCC and mutual informations."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from karios.matcher.mutual_info_service import MutualInfoService
from karios.matcher.patch_scores import compute_patch_scores
from karios.matcher.zncc_service import ZNCCService


def _image(array):
    return SimpleNamespace(array=array, x_size=array.shape[1], y_size=array.shape[0])


def _per_point(df, mon, ref):
    zncc, mi = ZNCCService(), MutualInfoService()
    return pd.DataFrame(
        {
            "zncc_score": df.apply(zncc._compute_zncc, axis=1, monitored=mon, reference=ref),
            "mutual_info_score": df.apply(
                mi._compute_mutual_info, axis=1, monitored=mon, reference=ref
            ),
            "mi_score": df.apply(zncc._compute_mi, axis=1, monitored=mon, reference=ref),
        }
    )


def _points(rng, n=500):
    df = pd.DataFrame(
        {
            "x0": rng.integers(0, 300, n).astype(float),
            "y0": rng.integers(0, 300, n).astype(float),
            "dx": rng.uniform(-3.5, 3.5, n),
            "dy": rng.uniform(-3.5, 3.5, n),
        }
    )
    # Inside the constant area below, with shifts of exactly half a pixel (rounded to even)
    df.loc[:9, ["x0", "y0"]] = 120.0
    df.loc[:9, ["dx", "dy"]] = 0.5
    return df


@pytest.fixture(name="rng")
def rng_fixture():
    return np.random.default_rng(0)


@pytest.mark.parametrize(
    "dtypes", [(np.uint16, np.uint16), (np.float32, np.float32), (np.uint16, np.float32)]
)
def test_scores_match_the_per_point_functions(rng, dtypes, caplog):
    caplog.set_level(logging.CRITICAL)
    base = rng.integers(0, 4000, (300, 300))
    ref = (base * (1 if dtypes[0] == np.uint16 else 0.37)).astype(dtypes[0])
    mon = (np.roll(base, 3, axis=1) * (1 if dtypes[1] == np.uint16 else 0.37)).astype(dtypes[1])
    ref[90:150, 90:150] = ref.flat[0]  # constant patches
    if mon.dtype.kind == "f":
        mon[200:210, 200:210] = np.nan  # non-finite patches
    df = _points(rng)

    got = compute_patch_scores(df, _image(mon), _image(ref))
    expected = _per_point(df, _image(mon), _image(ref))

    for column in ["mutual_info_score", "mi_score"]:
        np.testing.assert_allclose(got[column], expected[column], rtol=0, atol=1e-12)
    if dtypes == (np.uint16, np.uint16):
        np.testing.assert_allclose(got["zncc_score"], expected["zncc_score"], rtol=0, atol=1e-12)
    else:
        # ZNCC now in float64: a constant float32 patch gives NaN, where float32
        # rounding gave a non-zero deviation and a meaningless ~1e-8 score
        constant = got["zncc_score"].isna() & expected["zncc_score"].notna()
        assert (expected["zncc_score"][constant].abs() < 1e-6).all()
        keep = ~constant
        np.testing.assert_allclose(
            got["zncc_score"][keep], expected["zncc_score"][keep], rtol=0, atol=1e-6
        )


def test_points_leaving_either_image_are_nan(rng):
    array = rng.integers(0, 4000, (100, 100)).astype(np.uint16)
    df = pd.DataFrame(
        {"x0": [28.0, 27.0, 71.0, 50.0], "y0": [50.0] * 4, "dx": [0, 0, 0, 22.0], "dy": [0] * 4}
    )

    scores = compute_patch_scores(df, _image(array), _image(array))

    # 28 and 71 are the first and last centers whose 57 px patch fits; 50 + 22 leaves mon
    assert scores.notna().all(axis=1).tolist() == [True, False, True, False]


def test_empty_selection_gives_empty_scores(rng):
    array = rng.integers(0, 4000, (100, 100)).astype(np.uint16)
    df = pd.DataFrame(columns=["x0", "y0", "dx", "dy"], dtype=float)

    scores = compute_patch_scores(df, _image(array), _image(array))

    assert scores.empty and list(scores.columns) == ["zncc_score", "mutual_info_score", "mi_score"]
