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
from tests.utils.json_comparison_utils import compare_json_files
from tests.utils.test_csv_comparison import compare_csv_with_tolerance

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
            result_csv_path, ref_csv_path, float_tolerance=1e-5
        )

        if not comparison_result:
            print(f"\nCSV files differ: {diff_message}")
            self.fail(f"CSV files are different beyond tolerance. Details: {diff_message}")
        else:
            print(f"\nCSV files are equivalent within tolerance: {diff_message}")

        # Display first different lines of JSON/GeoJSON files in the log before the test
        json_result_path = os.path.join(
            result_dir,
            "L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04",
            geojson_filename,
        )
        json_ref_path = os.path.join(ref_data_dir, geojson_filename)

        print(f"Result JSON/GeoJSON content from: {json_result_path}")
        if os.path.exists(json_result_path) and os.path.exists(json_ref_path):
            with (
                open(json_result_path, "r") as f_result,
                open(json_ref_path, "r") as f_ref,
            ):
                result_json_lines = f_result.readlines()
                ref_json_lines = f_ref.readlines()

                # Find and display first 100 different lines
                diff_count = 0
                max_diffs_to_show = 100

                print(
                    f"Comparing files ({max(len(result_json_lines), len(ref_json_lines))} total lines), showing first {max_diffs_to_show} different lines:"
                )

                for i in range(min(len(result_json_lines), len(ref_json_lines))):
                    if result_json_lines[i] != ref_json_lines[i]:
                        if diff_count < max_diffs_to_show:
                            print(f"Line {i+1} - REF: {ref_json_lines[i].rstrip()}")
                            print(f"Line {i+1} - RES: {result_json_lines[i].rstrip()}")
                            diff_count += 1
                        else:
                            remaining_diffs = sum(
                                1
                                for j in range(i, min(len(result_json_lines), len(ref_json_lines)))
                                if result_json_lines[j] != ref_json_lines[j]
                            )
                            print(f"... and {remaining_diffs} more different lines")
                            break

                # Handle case where one file is longer than the other
                if len(result_json_lines) != len(ref_json_lines):
                    longer_file = (
                        result_json_lines
                        if len(result_json_lines) > len(ref_json_lines)
                        else ref_json_lines
                    )
                    shorter_file = (
                        ref_json_lines
                        if len(result_json_lines) > len(ref_json_lines)
                        else result_json_lines
                    )
                    start_idx = len(shorter_file)

                    for i in range(start_idx, len(longer_file)):
                        if diff_count < max_diffs_to_show:
                            file_type = (
                                "RES" if len(result_json_lines) > len(ref_json_lines) else "REF"
                            )
                            print(f"Line {i+1} - {file_type}: {longer_file[i].rstrip()}")
                            diff_count += 1
                        else:
                            print(
                                f"... and {len(longer_file) - start_idx - (max_diffs_to_show - diff_count)} more lines in the longer file"
                            )
                            break

                if diff_count == 0:
                    print("All compared lines are identical")

        elif os.path.exists(json_result_path):
            with open(json_result_path, "r") as f:
                result_json_lines = f.readlines()
                print(f"First 100 lines of result file ({len(result_json_lines)} total lines):")
                for i, line in enumerate(result_json_lines[:100]):
                    print(f"{i+1:3d}: {line.rstrip()}")
                if len(result_json_lines) > 100:
                    print(f"... and {len(result_json_lines) - 100} more lines")
        else:
            print(f"Result JSON/GeoJSON file does not exist: {json_result_path}")

        # Compare JSON files using proper JSON comparison with better error reporting
        json_comparison_result, json_diff_message = compare_json_files(
            json_result_path, json_ref_path
        )

        if not json_comparison_result:
            print(f"\nJSON files differ: {json_diff_message}")
            self.fail(f"JSON files are different. Details: {json_diff_message}")
        else:
            print(f"\nJSON files are equivalent: {json_diff_message}")
