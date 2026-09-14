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
"""Tests for the `align` output resolution fix in karios.matcher.global_align."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from osgeo import gdal

from karios.core.image import GdalRasterImage
from karios.matcher.global_align import GlobalAlignment, _output_geometry, apply_global_alignment


def _make_geotiff(path, array, x_min, y_max, x_res, y_res) -> GdalRasterImage:
    driver = gdal.GetDriverByName("GTiff")
    h, w = array.shape
    dataset = driver.Create(str(path), w, h, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform((x_min, x_res, 0, y_max, 0, y_res))
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    dataset = None
    return GdalRasterImage(str(path))


class TestOutputGeometry:
    """Unit tests for `_output_geometry`'s resolution/canvas-size math."""

    def test_monitored_finer_than_reference_upsamples_canvas(self):
        ref = SimpleNamespace(x_res=10.0, y_res=-10.0)
        mon = SimpleNamespace(x_res=5.0, y_res=-5.0)
        ow, oh, to_output = _output_geometry(ref, mon, rw=20, rh=15)
        assert (ow, oh) == (40, 30)
        np.testing.assert_allclose(to_output, np.diag([2.0, 2.0, 1.0]))

    def test_monitored_coarser_than_reference_downsamples_canvas(self):
        ref = SimpleNamespace(x_res=5.0, y_res=-5.0)
        mon = SimpleNamespace(x_res=10.0, y_res=-10.0)
        ow, oh, to_output = _output_geometry(ref, mon, rw=40, rh=30)
        assert (ow, oh) == (20, 15)
        np.testing.assert_allclose(to_output, np.diag([0.5, 0.5, 1.0]))

    def test_matching_resolution_is_unchanged(self):
        ref = SimpleNamespace(x_res=10.0, y_res=-10.0)
        mon = SimpleNamespace(x_res=10.0, y_res=-10.0)
        ow, oh, to_output = _output_geometry(ref, mon, rw=20, rh=15)
        assert (ow, oh) == (20, 15)
        np.testing.assert_allclose(to_output, np.eye(3))


class TestApplyGlobalAlignmentResolution:
    """`apply_global_alignment` must write outputs at mon's resolution, not ref's."""

    def test_outputs_use_monitored_resolution(self, tmp_path):
        ref_arr = (np.random.default_rng(0).random((20, 20)) * 255).astype(np.uint8)
        mon_arr = (np.random.default_rng(1).random((40, 40)) * 255).astype(np.uint8)

        reference = _make_geotiff(tmp_path / "ref.tif", ref_arr, 0.0, 0.0, 10.0, -10.0)
        monitored = _make_geotiff(tmp_path / "mon.tif", mon_arr, 0.0, 0.0, 5.0, -5.0)

        # mon → ref pixel-space homography implied by the geotransforms above
        # (same origin, mon pixels are half the size of ref pixels).
        matrix = np.array(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        fake_alignment = GlobalAlignment(matrix=matrix, n_inliers=10, n_matches=10, candidates=[])

        with patch(
            "karios.matcher.global_align.detect_global_alignment",
            return_value=fake_alignment,
        ):
            aligned_mon, ref_out, aligned_mask, alignment = apply_global_alignment(
                monitored, reference, None, tmp_path
            )

        assert alignment is fake_alignment
        assert aligned_mask is None

        # Output resolution must match the monitored image, not the reference.
        assert aligned_mon.x_res == monitored.x_res
        assert aligned_mon.y_res == monitored.y_res
        assert ref_out.x_res == monitored.x_res
        assert ref_out.y_res == monitored.y_res

        # Canvas covers ref's footprint (20x20 @ 10m) at mon's resolution (5m) -> 40x40.
        assert aligned_mon.array.shape == (40, 40)
        assert ref_out.array.shape == (40, 40)

        # The homography exactly cancels the resolution ratio, so mon's own
        # pixels pass through onto the output canvas unchanged.
        np.testing.assert_array_equal(aligned_mon.array, mon_arr)
