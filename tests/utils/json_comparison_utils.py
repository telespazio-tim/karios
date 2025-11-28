# -*- coding: utf-8 -*-
"""
Utility module for JSON comparison in tests.

This module provides functions to compare JSON objects and files,
with proper handling of object equality vs object identity.
"""

import json
from typing import Any, Dict, Union

import numpy as np


def compare_json_objects(
    result: Union[Dict, Any],
    ref: Union[Dict, Any],
    ignore_order: bool = False,
    float_tolerance: float = None,
) -> tuple[bool, str]:
    """
    Compare two JSON objects using proper equality (==) rather than identity (is).

    Args:
        result: The result JSON object to compare
        ref: The reference JSON object to compare against
        ignore_order: Whether to ignore order in lists/arrays
        float_tolerance: Tolerance for floating-point comparison (default: None, which means exact comparison)

    Returns:
        tuple: (comparison_result, diff_message)
            - comparison_result: True if objects are equal, False otherwise
            - diff_message: Detailed message about differences or success
    """
    # Use deep comparison with optional float tolerance
    if float_tolerance is not None:
        are_equal, diff_msg = compare_json_with_tolerance_and_stats(result, ref, float_tolerance)
        return are_equal, diff_msg
    elif ignore_order:
        # For comparing JSON with potentially different ordering
        try:
            result_sorted = _sort_json_objects(result)
            ref_sorted = _sort_json_objects(ref)
            are_equal = result_sorted == ref_sorted
        except Exception as e:
            return False, f"Error sorting JSON objects for comparison: {str(e)}"
    else:
        are_equal = result == ref

    if are_equal:
        return True, "JSON objects are equivalent"
    else:
        # Try to provide more detailed information about the difference
        try:
            result_str = json.dumps(result, sort_keys=True, indent=2)
            ref_str = json.dumps(ref, sort_keys=True, indent=2)
            return (
                False,
                f"JSON objects differ:\nReference:\n{ref_str}\n\nResult:\n{result_str}",
            )
        except Exception:
            # If JSON serialization fails, provide basic error message
            return False, "JSON objects differ"


def compare_json_files(
    result_json_path: str,
    ref_json_path: str,
    ignore_order: bool = False,
    float_tolerance: float = None,
) -> tuple[bool, str]:
    """
    Compare two JSON files for equality.

    Args:
        result_json_path: Path to the result JSON file
        ref_json_path: Path to the reference JSON file
        ignore_order: Whether to ignore order in lists/arrays
        float_tolerance: Tolerance for floating-point comparison (default: None, which means exact comparison)

    Returns:
        tuple: (comparison_result, diff_message)
    """
    try:
        with open(result_json_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        with open(ref_json_path, "r", encoding="utf-8") as f:
            ref = json.load(f)

        return compare_json_objects(result, ref, ignore_order, float_tolerance)

    except FileNotFoundError as e:
        return False, f"File not found: {str(e)}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in file: {str(e)}"
    except Exception as e:
        return False, f"Error comparing JSON files: {str(e)}"


def _compare_json_with_tolerance(
    obj1: Any, obj2: Any, tolerance: float, diff_values=None
) -> tuple[bool, str]:
    """
    Recursively compare JSON objects with float tolerance.

    Args:
        obj1: First JSON object to compare
        obj2: Second JSON object to compare
        tolerance: Tolerance for floating point comparison
        diff_values: List to accumulate float differences (for statistics)

    Returns:
        tuple: (comparison_result, diff_message)
    """
    if diff_values is None:
        diff_values = []

    if type(obj1) != type(obj2):
        return False, f"Different types: {type(obj1).__name__} vs {type(obj2).__name__}"

    if isinstance(obj1, float) and isinstance(obj2, float):
        abs_diff = abs(obj1 - obj2)
        diff_values.append(abs_diff)
        if abs_diff <= tolerance:
            return True, ""
        else:
            return (
                False,
                f"Float values differ beyond tolerance: {obj1} vs {obj2} (diff: {abs_diff}, tolerance: {tolerance})",
            )
    elif isinstance(obj1, (int, str, type(None))):
        if obj1 == obj2:
            return True, ""
        else:
            return False, f"Values differ: {obj1} vs {obj2}"
    elif isinstance(obj1, dict) and isinstance(obj2, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            return False, f"Different keys: {set(obj1.keys())} vs {set(obj2.keys())}"

        for key in obj1.keys():
            is_equal, msg = _compare_json_with_tolerance(
                obj1[key], obj2[key], tolerance, diff_values
            )
            if not is_equal:
                return False, f"Key '{key}' differs: {msg}"
        return True, ""
    elif isinstance(obj1, list) and isinstance(obj2, list):
        if len(obj1) != len(obj2):
            return False, f"Different list lengths: {len(obj1)} vs {len(obj2)}"

        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            is_equal, msg = _compare_json_with_tolerance(item1, item2, tolerance, diff_values)
            if not is_equal:
                return False, f"Item at index {i} differs: {msg}"
        return True, ""
    else:
        if obj1 == obj2:
            return True, ""
        else:
            return False, f"Objects differ: {obj1} vs {obj2}"


def compare_json_with_tolerance_and_stats(
    obj1: Any, obj2: Any, tolerance: float
) -> tuple[bool, str]:
    """
    Compare JSON objects with float tolerance and return statistics on differences.

    Args:
        obj1: First JSON object to compare
        obj2: Second JSON object to compare
        tolerance: Tolerance for floating point comparison

    Returns:
        tuple: (comparison_result, diff_message)
    """
    diff_values = []
    is_equal, msg = _compare_json_with_tolerance(obj1, obj2, tolerance, diff_values)

    if is_equal and len(diff_values) > 0:
        # Calculate statistics for the differences
        std_diff = np.std(diff_values)
        min_diff = np.min(diff_values)
        max_diff = np.max(diff_values)
        mean_diff = np.mean(diff_values)

        stats_msg = f"JSON objects are equivalent within tolerance {tolerance}. Statistics: std={std_diff:.2e}, min={min_diff:.2e}, max={max_diff:.2e}, mean={mean_diff:.2e} (n={len(diff_values)})"
        return True, stats_msg
    elif is_equal:
        # No differences found
        return True, f"JSON objects are equivalent within tolerance {tolerance}"
    else:
        # If not equal, return the original message without statistics
        return False, msg


def _sort_json_objects(obj: Any) -> Any:
    """
    Recursively sort JSON objects to normalize ordering for comparison.

    Args:
        obj: JSON object to sort

    Returns:
        Sorted version of the object
    """
    if isinstance(obj, dict):
        return {k: _sort_json_objects(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        # Try to sort if all elements are comparable, otherwise return as-is
        try:
            # If all elements are the same type and comparable, sort them
            if obj and all(isinstance(item, (str, int, float)) for item in obj):
                return sorted(obj)
            else:
                # For complex objects, sort based on JSON serialization
                return [_sort_json_objects(item) for item in obj]
        except TypeError:
            # If elements are not comparable, return as-is but with nested items sorted
            return [_sort_json_objects(item) for item in obj]
    else:
        return obj
