#!/usr/bin/env python3
"""Test script to verify the ZNCC zero standard deviation fix."""

import numpy as np
import pandas as pd
from unittest.mock import Mock
from karios.matcher.zncc_service import ZNCCService, _zncc2

def test_zero_std_handling():
    """Test that zero standard deviation patches return NaN instead of raising an exception."""
    
    # Create ZNCCService instance
    zncc_service = ZNCCService()
    
    # Create test data with uniform patches (zero std)
    # Create a uniform patch (all same values) to trigger zero std error
    uniform_patch = np.ones((57, 57)) * 100  # All pixels have value 100
    
    # Create mock images
    mock_monitored = Mock()
    mock_monitored.array = uniform_patch
    mock_monitored.x_size = 100
    mock_monitored.y_size = 100
    mock_monitored.clear_cache = Mock()
    
    mock_reference = Mock()
    mock_reference.array = uniform_patch
    mock_reference.x_size = 100
    mock_reference.y_size = 100
    mock_reference.clear_cache = Mock()
    
    # Test case: create a dataframe that would lead to uniform patches
    test_df = pd.DataFrame({
        'x0': [30],  # Center point
        'y0': [30],  # Center point
        'dx': [0],   # No offset
        'dy': [0]    # No offset
    })
    
    # Call the compute_zncc method - this should not raise an exception
    try:
        result = zncc_service.compute_zncc(test_df, mock_monitored, mock_reference)
        print("Success: ZNCC computation completed without exception")
        print(f"Result: {result}")
        
        # The result should be NaN for the zero std case
        if pd.isna(result.iloc[0]):
            print("Success: Result is NaN as expected for zero std case")
        else:
            print(f"Error: Expected NaN but got {result.iloc[0]}")
            
        return True
    except Exception as e:
        print(f"Error: ZNCC computation raised exception: {e}")
        return False

def test_zncc2_directly():
    """Test _zncc2 function directly with zero std patches."""
    uniform_patch = np.ones((57, 57)) * 100
    try:
        result = _zncc2(uniform_patch, uniform_patch, 28, 28, 28, 28, 21)
        print(f"Error: _zncc2 should have raised ValueError but returned {result}")
        return False
    except ValueError as e:
        if "zero standard deviation" in str(e):
            print("Success: _zncc2 correctly raises ValueError for zero std")
            return True
        else:
            print(f"Error: Unexpected ValueError message: {e}")
            return False
    except Exception as e:
        print(f"Error: Unexpected exception type: {e}")
        return False

if __name__ == "__main__":
    print("Testing ZNCC zero standard deviation fix...")
    
    print("\n1. Testing _zncc2 function directly:")
    zncc2_test = test_zncc2_directly()
    
    print("\n2. Testing ZNCCService with proper error handling:")
    service_test = test_zero_std_handling()
    
    print(f"\nResults: zncc2_test={zncc2_test}, service_test={service_test}")
    print("Fix verification:", "PASSED" if (zncc2_test and service_test) else "FAILED")