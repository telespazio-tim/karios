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


def normalize_csv_content(content):
    """Normalize CSV content by stripping whitespace and normalizing line endings."""
    lines = content.splitlines()
    normalized_lines = []
    for line in lines:
        # Strip leading/trailing whitespace and normalize internal whitespace
        normalized_line = ';'.join(field.strip() for field in line.split(';'))
        normalized_lines.append(normalized_line)
    return normalized_lines


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

        # Compare CSV files with diff display
        result_csv_path = os.path.join(
            result_dir,
            "L2F_T12SYH_20220824T175017_LS9_R035_B04_10m_T12SYH_20220514T175909_B04",
            csv_result_filename,
        )
        ref_csv_path = os.path.join(ref_data_dir, csv_result_filename)

        # Read and normalize both CSV files
        with open(result_csv_path, 'r') as f:
            result_csv_lines = normalize_csv_content(f.read())

        with open(ref_csv_path, 'r') as f:
            ref_csv_lines = normalize_csv_content(f.read())

        # Check if files are different and show diff if they are
        if result_csv_lines != ref_csv_lines:
            print("\nCSV files are different. Showing diff (ignoring whitespace):")
            diff = difflib.unified_diff(
                ref_csv_lines,
                result_csv_lines,
                fromfile=f"ref: {ref_csv_path}",
                tofile=f"res: {result_csv_path}",
                lineterm=''
            )
            diff_text = '\n'.join(diff)
            print(diff_text)
            self.fail("CSV files are different. See diff above.")
        else:
            print("\nCSV files are identical (ignoring whitespace differences).")

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
