# -*- coding: utf-8 -*-
"""
Utility module for JSON comparison in tests.

This module provides functions to compare JSON objects and files,
with proper handling of object equality vs object identity.
"""

import json
from typing import Any, Dict, Union


def compare_json_objects(result: Union[Dict, Any], ref: Union[Dict, Any], ignore_order: bool = False) -> tuple[bool, str]:
    """
    Compare two JSON objects using proper equality (==) rather than identity (is).

    Args:
        result: The result JSON object to compare
        ref: The reference JSON object to compare against
        ignore_order: Whether to ignore order in lists/arrays

    Returns:
        tuple: (comparison_result, diff_message)
            - comparison_result: True if objects are equal, False otherwise
            - diff_message: Detailed message about differences or success
    """
    # Use == for proper value comparison, not 'is' which checks object identity
    if ignore_order:
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
            return False, f"JSON objects differ:\nReference:\n{ref_str}\n\nResult:\n{result_str}"
        except Exception:
            # If JSON serialization fails, provide basic error message
            return False, "JSON objects differ"


def compare_json_files(result_json_path: str, ref_json_path: str, ignore_order: bool = False) -> tuple[bool, str]:
    """
    Compare two JSON files for equality.

    Args:
        result_json_path: Path to the result JSON file
        ref_json_path: Path to the reference JSON file
        ignore_order: Whether to ignore order in lists/arrays

    Returns:
        tuple: (comparison_result, diff_message)
    """
    try:
        with open(result_json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        with open(ref_json_path, 'r', encoding='utf-8') as f:
            ref = json.load(f)
        
        return compare_json_objects(result, ref, ignore_order)
    
    except FileNotFoundError as e:
        return False, f"File not found: {str(e)}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in file: {str(e)}"
    except Exception as e:
        return False, f"Error comparing JSON files: {str(e)}"


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