# -*- coding: utf-8 -*-
"""
Utility module for tolerance-based CSV comparison in end-to-end tests.

This module provides functions to compare CSV files with floating-point values
that may have small platform-dependent differences due to compiler optimizations
or different underlying math libraries.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def compare_csv_with_tolerance(
    result_csv_path: str,
    ref_csv_path: str,
    float_tolerance: float = 1e-5,
    separator: str = ";"
) -> tuple[bool, str]:
    """
    Compare two CSV files with tolerance for floating-point differences.
    
    Args:
        result_csv_path: Path to the result CSV file
        ref_csv_path: Path to the reference CSV file
        float_tolerance: Tolerance for floating-point comparison (default: 1e-5)
        separator: CSV separator (default: ";")
        
    Returns:
        tuple: (comparison_result, diff_message)
            - comparison_result: True if files are considered equal, False otherwise
            - diff_message: Detailed message about differences or success
    """
    try:
        # Read both CSV files
        ref_df = pd.read_csv(ref_csv_path, sep=separator, dtype=str)
        result_df = pd.read_csv(result_csv_path, sep=separator, dtype=str)
        
        # Check if shapes are different
        if ref_df.shape != result_df.shape:
            return False, f"CSV files have different shapes: ref {ref_df.shape} vs result {result_df.shape}"
        
        # Convert columns that are purely numeric to float, others remain as string
        ref_df_processed = _convert_numeric_columns(ref_df)
        result_df_processed = _convert_numeric_columns(result_df)
        
        # Check column names
        if not ref_df_processed.columns.equals(result_df_processed.columns):
            return False, f"Column names differ: ref {ref_df_processed.columns.tolist()} vs result {result_df_processed.columns.tolist()}"
        
        # Compare non-numeric columns exactly
        non_numeric_cols = []
        numeric_cols = []
        
        for col in ref_df_processed.columns:
            ref_is_numeric = pd.api.types.is_numeric_dtype(ref_df_processed[col])
            res_is_numeric = pd.api.types.is_numeric_dtype(result_df_processed[col])
            
            if ref_is_numeric and res_is_numeric:
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)
        
        # Check non-numeric columns for exact match
        for col in non_numeric_cols:
            if not ref_df_processed[col].equals(result_df_processed[col]):
                # Find the first differing row
                diff_mask = ref_df_processed[col] != result_df_processed[col]
                first_diff_idx = diff_mask.idxmax()
                return False, f"Non-numeric column '{col}' differs at row {first_diff_idx}: ref='{ref_df_processed[col][first_diff_idx]}', result='{result_df_processed[col][first_diff_idx]}'"
        
        # Compare numeric columns with tolerance
        for col in numeric_cols:
            ref_series = pd.to_numeric(ref_df_processed[col], errors='coerce')
            res_series = pd.to_numeric(result_df_processed[col], errors='coerce')
            
            # Check for NaN values
            ref_nan_mask = ref_series.isna()
            res_nan_mask = res_series.isna()
            
            if not ref_nan_mask.equals(res_nan_mask):
                return False, f"Different NaN patterns in column '{col}'"
            
            # Compare non-NaN values with tolerance
            valid_mask = ~ref_nan_mask
            if valid_mask.any():
                ref_values = ref_series[valid_mask]
                res_values = res_series[valid_mask]
                
                # Use numpy's allclose for element-wise comparison with tolerance
                if not np.allclose(ref_values, res_values, rtol=float_tolerance, atol=float_tolerance, equal_nan=True):
                    # Find and report the first significant difference
                    abs_diff = np.abs(ref_values - res_values)
                    max_diff_idx = abs_diff.argmax()
                    max_diff = abs_diff.max()
                    max_ref_val = ref_values.iloc[max_diff_idx]
                    max_res_val = res_values.iloc[max_diff_idx]
                    
                    return False, f"Numeric column '{col}' differs beyond tolerance at row {max_diff_idx}: ref={max_ref_val}, result={max_res_val}, diff={max_diff}, tolerance={float_tolerance}"
        
        return True, "CSV files are equivalent within tolerance"
        
    except Exception as e:
        return False, f"Error comparing CSV files: {str(e)}"


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to convert columns to numeric where possible, leaving non-numeric columns unchanged.
    
    Args:
        df: Input DataFrame with string values
        
    Returns:
        DataFrame with numeric columns converted to float where possible
    """
    df_converted = df.copy()
    
    for col in df.columns:
        # Try to convert to numeric, leaving non-numeric values as-is
        converted_series = pd.to_numeric(df[col], errors='coerce')
        
        # If conversion introduced NaN for valid values, keep original string
        if converted_series.isna().any() and not df[col].isna().any():
            # Check if any non-numeric values were converted to NaN
            numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
            original_mask = df[col].notna()
            if not numeric_mask.equals(original_mask):
                # Keep as string since some values couldn't be converted to numeric
                continue
        else:
            # All values in column were converted successfully or were already NaN
            df_converted[col] = converted_series
    
    return df_converted


def compare_csv_files_exact(result_csv_path: str, ref_csv_path: str, separator: str = ";") -> tuple[bool, str]:
    """
    Traditional exact comparison of two CSV files.
    
    Args:
        result_csv_path: Path to the result CSV file
        ref_csv_path: Path to the reference CSV file
        separator: CSV separator (default: ";")
        
    Returns:
        tuple: (comparison_result, diff_message)
    """
    with open(result_csv_path, 'r') as f:
        result_lines = f.read().strip().split('\n')
    
    with open(ref_csv_path, 'r') as f:
        ref_lines = f.read().strip().split('\n')
    
    if result_lines == ref_lines:
        return True, "CSV files are exactly identical"
    else:
        # Find first difference
        max_len = max(len(result_lines), len(ref_lines))
        for i in range(max_len):
            if i >= len(ref_lines):
                return False, f"Result file has more lines. First extra line at {i+1}: {result_lines[i]}"
            if i >= len(result_lines):
                return False, f"Reference file has more lines. First extra line at {i+1}: {ref_lines[i]}"
            if result_lines[i] != ref_lines[i]:
                return False, f"First difference at line {i+1}: ref='{ref_lines[i]}', result='{result_lines[i]}'"
        
        return False, "Files differ (detailed diff not generated)"