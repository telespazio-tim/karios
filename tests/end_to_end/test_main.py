# -*- coding: utf-8 -*-
"""End 2 end test module"""
import difflib
import filecmp
import os
import shutil
import unittest
from pathlib import Path

from click.testing import CliRunner

from karios.cli.commands import process
from tests.utils.test_csv_comparison import compare_csv_with_tolerance
from tests.utils.json_comparison_utils import compare_json_files




module_dir_path = os.path.dirname(__file__)
result_dir = os.path.join(module_dir_path, "test_results")
result_dir_path = Path(result_dir)
ref_data_dir = os.path.join(module_dir_path, "ref_data", "test_full")
test_data_dir = os.path.join(module_dir_path, "test_data")


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
        """Test full processing"""
        csv_result_filename = (
            "KLT_matcher_L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04.csv"
        )
        geojson_filename = "kp_delta.json"

        runner = CliRunner()
        result = runner.invoke(
            process,
            [
                os.path.join(test_data_dir, "L2F_T12SYH_20220824T175017_LS9_R035_B04_10m.tif"),
                os.path.join(test_data_dir, "T12SYH_20220514T175909_B04.tif"),
                "--out",
                result_dir,
                "--conf",
                os.path.join(module_dir_path, "processing_configuration.json"),
            ],
        )

        self.assertEqual(0, result.exit_code)

        # Display both CSV files in the log before the test
        result_csv_path = os.path.join(
            result_dir,
            "L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04",
            csv_result_filename,
        )
        ref_csv_path = os.path.join(ref_data_dir, csv_result_filename)

        print(f"Result CSV content from: {result_csv_path}")
        if os.path.exists(result_csv_path):
            with open(result_csv_path, "r") as f:
                result_csv_content = f.read()
                print(
                    result_csv_content[:2000] + "..."
                    if len(result_csv_content) > 2000
                    else result_csv_content
                )  # Limit output to first 2000 characters
        else:
            print(f"Result CSV file does not exist: {result_csv_path}")

        print(f"\nReference CSV content from: {ref_csv_path}")
        if os.path.exists(ref_csv_path):
            with open(ref_csv_path, "r") as f:
                ref_csv_content = f.read()
                print(
                    ref_csv_content[:2000] + "..."
                    if len(ref_csv_content) > 2000
                    else ref_csv_content
                )  # Limit output to first 2000 characters
        else:
            print(f"Reference CSV file does not exist: {ref_csv_path}")

        # Compare CSV files with tolerance for floating-point differences
        result_csv_path = os.path.join(
            result_dir,
            "L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04",
            csv_result_filename,
        )
        ref_csv_path = os.path.join(ref_data_dir, csv_result_filename)

        # Use tolerance-based comparison for CSV files
        comparison_result, diff_message = compare_csv_with_tolerance(
            result_csv_path,
            ref_csv_path,
            float_tolerance=1e-5
        )

        if not comparison_result:
            print(f"\nCSV files differ: {diff_message}")
            self.fail(f"CSV files are different beyond tolerance. Details: {diff_message}")
        else:
            print(f"\nCSV files are equivalent within tolerance: {diff_message}")

        # Compare JSON files using proper JSON comparison with better error reporting
        json_result_path = os.path.join(
            result_dir,
            "L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04",
            geojson_filename,
        )
        json_ref_path = os.path.join(ref_data_dir, geojson_filename)

        json_comparison_result, json_diff_message = compare_json_files(
            json_result_path,
            json_ref_path
        )

        if not json_comparison_result:
            print(f"\nJSON files differ: {json_diff_message}")
            self.fail(f"JSON files are different. Details: {json_diff_message}")
        else:
            print(f"\nJSON files are equivalent: {json_diff_message}")
