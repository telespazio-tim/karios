# -*- coding: utf-8 -*-
# Copyright (c) 2025 Telespazio France.
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
"""Modules for large offset matcher"""

from skimage.registration import phase_cross_correlation

from core.image import GdalRasterImage


class LargeOffsetMatcher:
    """Class to compute row/col offset between 2 images"""

    def __init__(self, reference_image: GdalRasterImage, monitored_image: GdalRasterImage):
        self._ref = reference_image
        self._mon = monitored_image

    def match(self):
        """Computes row/col offset between 2 images

        Returns:
            [int, int]: row/col (y/x) offset
        """
        # ref and mon parameters are deliberately inverted
        return phase_cross_correlation(self._mon.array, self._ref.array)[
            0
        ]  # , reference_mask=reference_mask, moving_mask=moving_mask)
