#!/usr/bin/env python3
"""Unit tests for LargeOffsetMatcher class."""

import numpy as np
from unittest.mock import Mock, patch
from karios.matcher.large_offset import LargeOffsetMatcher
from karios.core.image import GdalRasterImage


def test_large_offset_matcher_initialization():
    """Test LargeOffsetMatcher initialization."""
    mock_ref_image = Mock(spec=GdalRasterImage)
    mock_mon_image = Mock(spec=GdalRasterImage)
    
    matcher = LargeOffsetMatcher(mock_ref_image, mock_mon_image)
    
    assert matcher._ref == mock_ref_image
    assert matcher._mon == mock_mon_image


@patch('karios.matcher.large_offset.phase_cross_correlation')
def test_large_offset_matcher_match(mock_phase_corr):
    """Test LargeOffsetMatcher match method."""
    mock_ref_image = Mock(spec=GdalRasterImage)
    mock_mon_image = Mock(spec=GdalRasterImage)
    
    # Create test arrays
    ref_array = np.random.rand(100, 100)
    mon_array = np.random.rand(100, 100)
    
    mock_ref_image.array = ref_array
    mock_mon_image.array = mon_array
    
    # Mock the phase cross correlation result
    mock_offset = np.array([5.5, 3.2])  # [row_offset, col_offset] or [y_offset, x_offset]
    mock_phase_corr.return_value = (mock_offset,)
    
    matcher = LargeOffsetMatcher(mock_ref_image, mock_mon_image)
    result = matcher.match()
    
    # Verify the phase_cross_correlation was called with the right arguments
    # Note: In the code, ref and mon parameters are deliberately inverted
    mock_phase_corr.assert_called_once_with(mon_array, ref_array)
    
    # Verify the result matches the offset returned by the mocked function
    assert np.array_equal(result, mock_offset)


@patch('karios.matcher.large_offset.phase_cross_correlation')
def test_large_offset_matcher_match_with_different_arrays(mock_phase_corr):
    """Test LargeOffsetMatcher with different arrays to see different offsets."""
    mock_ref_image = Mock(spec=GdalRasterImage)
    mock_mon_image = Mock(spec=GdalRasterImage)
    
    # Create different test arrays
    ref_array = np.ones((50, 50))
    mon_array = np.ones((50, 50)) * 2
    
    mock_ref_image.array = ref_array
    mock_mon_image.array = mon_array
    
    # Mock a different offset
    mock_offset = np.array([-2.1, 4.7])
    mock_phase_corr.return_value = (mock_offset,)
    
    matcher = LargeOffsetMatcher(mock_ref_image, mock_mon_image)
    result = matcher.match()
    
    assert np.array_equal(result, mock_offset)


@patch('karios.matcher.large_offset.phase_cross_correlation')
def test_large_offset_matcher_match_zero_offset(mock_phase_corr):
    """Test LargeOffsetMatcher with zero offset."""
    mock_ref_image = Mock(spec=GdalRasterImage)
    mock_mon_image = Mock(spec=GdalRasterImage)
    
    ref_array = np.random.rand(64, 64)
    mon_array = ref_array.copy()  # Same array so offset should be zero
    
    mock_ref_image.array = ref_array
    mock_mon_image.array = mon_array
    
    # Mock zero offset
    mock_offset = np.array([0.0, 0.0])
    mock_phase_corr.return_value = (mock_offset,)
    
    matcher = LargeOffsetMatcher(mock_ref_image, mock_mon_image)
    result = matcher.match()
    
    assert np.array_equal(result, mock_offset)


@patch('karios.matcher.large_offset.phase_cross_correlation')
def test_large_offset_matcher_match_negative_offset(mock_phase_corr):
    """Test LargeOffsetMatcher with negative offset."""
    mock_ref_image = Mock(spec=GdalRasterImage)
    mock_mon_image = Mock(spec=GdalRasterImage)
    
    ref_array = np.random.rand(100, 100)
    mon_array = np.random.rand(100, 100)
    
    mock_ref_image.array = ref_array
    mock_mon_image.array = mon_array
    
    # Mock negative offset
    mock_offset = np.array([-3.5, -2.1])
    mock_phase_corr.return_value = (mock_offset,)
    
    matcher = LargeOffsetMatcher(mock_ref_image, mock_mon_image)
    result = matcher.match()
    
    assert np.array_equal(result, mock_offset)


def test_large_offset_matcher_attributes():
    """Test that LargeOffsetMatcher properly stores its attributes."""
    mock_ref_image = Mock(spec=GdalRasterImage)
    mock_mon_image = Mock(spec=GdalRasterImage)
    
    matcher = LargeOffsetMatcher(mock_ref_image, mock_mon_image)
    
    # Verify attributes are stored correctly
    assert hasattr(matcher, '_ref')
    assert hasattr(matcher, '_mon')
    assert matcher._ref == mock_ref_image
    assert matcher._mon == mock_mon_image


if __name__ == "__main__":
    test_large_offset_matcher_initialization()
    test_large_offset_matcher_match()
    test_large_offset_matcher_match_with_different_arrays()
    test_large_offset_matcher_match_zero_offset()
    test_large_offset_matcher_match_negative_offset()
    test_large_offset_matcher_attributes()
    print("All LargeOffsetMatcher tests passed!")