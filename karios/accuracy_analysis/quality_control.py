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
"""Confidence that the matching measured a real misregistration.

KLT returns key points even between two unrelated images, and the statistics
computed on them then look like a measurement. Three indicators tell the cases
apart, each by an order of magnitude on 2000 px crops of a Landsat-9 / Sentinel-2
pair against the same monitored crop on another area, offset by 30 or 300 px,
or flipped:

- the median ZNCC of the confident points' patches: 0.88 related, 0.03-0.06
  unrelated, about 0 for patches paired at random;
- the share of confident points moving like their neighbours, within
  COHERENCE_PX of the median shift of the COHERENCE_NEIGHBOURS nearest ones:
  0.97 related, 0.05-0.06 unrelated, where KLT locks onto random shifts;
- the share of detected corners kept by KLT's forward-backward check: 36%
  related, 3.5% unrelated.

A related pair shifted beyond KLT's reach (8 px with the default pyramid) scores
as low as an unrelated one: its statistics are just as meaningless.

Each indicator is mapped to [0, 1] by a linear ramp from the level of unrelated
images to the level of related ones, and the confidence is the mean of the
indicators available: no ZNCC when a large shift was applied, no tracking ratio
when the key points were reloaded with --resume.
"""

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)

RELIABLE = "reliable"
DOUBTFUL = "doubtful"
UNRELIABLE = "unreliable"

# (unrelated, related) levels of each indicator, the ramp's ends
ZNCC_RAMP = (0.1, 0.5)
COHERENCE_RAMP = (0.2, 0.7)
TRACKING_RAMP = (0.05, 0.2)
COHERENCE_PX = 1.0
COHERENCE_NEIGHBOURS = 8
# Confidence at or above which the matching is reliable, below which it is not
RELIABLE_CONFIDENCE = 0.7
UNRELIABLE_CONFIDENCE = 0.3
# Fewer confident points than this measure nothing
MIN_CONFIDENT_POINTS = 30

UNRELATED_HINT = (
    "the images may be unrelated, or misregistered beyond the matching range "
    "(see --enable-large-shift-detection)"
)


@dataclass
class QualityControl:
    """Confidence in the matching result, and the indicators it comes from."""

    confidence: float  # in [0, 1]
    verdict: str  # RELIABLE, DOUBTFUL or UNRELIABLE
    detected_points: Optional[int]  # corners KLT tracked, None when not known (--resume)
    tracked_points: int  # key points kept by the forward-backward check
    confident_points: int  # key points above the confidence threshold
    tracking_ratio: Optional[float]  # tracked / detected
    median_zncc: Optional[float]  # of the confident points' patches
    coherent_fraction: Optional[float]  # of the confident points moving like their neighbours
    reasons: list = field(default_factory=list)  # why the confidence is not full

    def to_dict(self) -> dict:
        """The fields, with None for undefined values: JSON has no NaN."""
        return {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in asdict(self).items()
        }


def _ramp(value: float, ends: tuple[float, float]) -> float:
    low, high = ends
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _coherent_fraction(points: DataFrame) -> float:
    """Share of `points` whose shift is within COHERENCE_PX of their neighbours' median."""
    from scipy.spatial import cKDTree  # pylint: disable=import-outside-toplevel

    positions = points[["x0", "y0"]].to_numpy(dtype=float)
    shifts = points[["dx", "dy"]].to_numpy(dtype=float)
    k = min(COHERENCE_NEIGHBOURS, len(points) - 1)
    # The nearest point is the point itself
    _, neighbours = cKDTree(positions).query(positions, k=k + 1)
    local = np.median(shifts[neighbours[:, 1:]], axis=1)
    deviation = np.hypot(*(shifts - local).T)
    return float(np.mean(deviation <= COHERENCE_PX))


def assess_quality(
    points: DataFrame, confidence_threshold: float, detected_points: Optional[int]
) -> QualityControl:
    """Confidence that `points` measure a real misregistration, see the module docstring.

    Args:
        points: KLT key points, with columns x0, y0, dx, dy, score and, when
            computed, zncc_score.
        confidence_threshold: KLT score above which a point is confident, as for
            the accuracy statistics.
        detected_points: corners KLT tracked, None when unknown.
    """
    confident = points[points["score"] > confidence_threshold]
    tracked = len(points)
    tracking_ratio = tracked / detected_points if detected_points else None
    if len(confident) < MIN_CONFIDENT_POINTS:
        reason = (
            f"only {len(confident)} key points above the confidence threshold "
            f"{confidence_threshold} (at least {MIN_CONFIDENT_POINTS} needed)"
        )
        return QualityControl(
            0.0,
            UNRELIABLE,
            detected_points,
            tracked,
            len(confident),
            tracking_ratio,
            None,
            None,
            [reason, UNRELATED_HINT],
        )

    median_zncc = None
    if "zncc_score" in confident and confident["zncc_score"].notna().any():
        median_zncc = float(np.nanmedian(confident["zncc_score"]))
    coherent_fraction = _coherent_fraction(confident)

    indicators = [
        ("median ZNCC", median_zncc, ZNCC_RAMP, "unrelated images give about 0"),
        (
            "coherent key points",
            coherent_fraction,
            COHERENCE_RAMP,
            "share moving within 1 px of their neighbours, unrelated images give about 0.05",
        ),
        (
            "tracking ratio",
            tracking_ratio,
            TRACKING_RAMP,
            "share of detected corners kept, unrelated images give about 0.035",
        ),
    ]
    scores, reasons = [], []
    for name, value, ends, meaning in indicators:
        if value is None:
            continue
        scores.append(_ramp(value, ends))
        if value < ends[1]:
            reasons.append(f"{name} {value:.2f} below {ends[1]} ({meaning})")
    confidence = float(np.mean(scores))

    if confidence >= RELIABLE_CONFIDENCE:
        verdict = RELIABLE
    elif confidence < UNRELIABLE_CONFIDENCE:
        verdict = UNRELIABLE
        reasons.append(UNRELATED_HINT)
    else:
        verdict = DOUBTFUL
    return QualityControl(
        confidence,
        verdict,
        detected_points,
        tracked,
        len(confident),
        tracking_ratio,
        median_zncc,
        coherent_fraction,
        reasons,
    )
