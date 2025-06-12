# -*- coding: utf-8 -*-
"""End 2 end test module"""
import filecmp
import os
import shutil
import unittest
from pathlib import Path

from click.testing import CliRunner

from karios.cli.commands import process

module_dir_path = os.path.dirname(__file__)
result_dir = os.path.join(module_dir_path, "test_results")
result_dir_path = Path(result_dir)
ref_data_dir = os.path.join(module_dir_path, "ref_data", "test_full")
config_file_path = os.path.join(module_dir_path, "processing_configuration.json")


class E2ETest(unittest.TestCase):
    def setUp(self):
        if not result_dir_path.exists():
            print("Create %s", result_dir_path)
            result_dir_path.mkdir()

    def tearDown(self):
        if result_dir_path.exists():
            print("Remove %s", result_dir_path)
            shutil.rmtree(result_dir_path)

    def test_full(self):
        csv_result_filename = (
            "KLT_matcher_L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04.csv"
        )
        geojson_filename = "kp_delta.json"

        runner = CliRunner()
        result = runner.invoke(
            process,
            [
                "/media/pcanonici/KINGSTON/data/KARIOS/monitored/L2F_T12SYH_20220824T175017_LS9_R035_B04_10m.TIF",
                "/media/pcanonici/KINGSTON/data/KARIOS/ref/T12SYH_20220514T175909_B04.jp2",
                "--out",
                result_dir,
                "--conf",
                config_file_path,
            ],
        )

        self.assertEqual(0, result.exit_code)

        self.assertTrue(
            filecmp.cmp(
                os.path.join(
                    result_dir,
                    "L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04",
                    csv_result_filename,
                ),
                os.path.join(ref_data_dir, csv_result_filename),
                False,
            )
        )

        self.assertTrue(
            filecmp.cmp(
                os.path.join(
                    result_dir,
                    "L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04",
                    geojson_filename,
                ),
                os.path.join(ref_data_dir, geojson_filename),
                False,
            )
        )
