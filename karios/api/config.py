# -*- coding: utf-8 -*-
# Copyright (c) 2025 Telespazio France.
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
"""Runtime configuration module for KARIOS.

This module defines the RuntimeConfiguration dataclass which specifies how
KARIOS should process images, but not which specific images to process.
This separation allows the same configuration to be reused across multiple
image processing operations.

The configuration covers:
- Output settings (directory, file generation flags)
- Processing parameters (pixel size, large shift detection)
- Optional inputs (mask files, DEM files)
- Visualization options (title prefixes, DEM descriptions)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RuntimeConfiguration:
    """Runtime configuration for KARIOS processing behavior.

    This configuration defines how images should be processed, focusing purely
    on processing behavior and output settings. Input files (images, masks, DEMs)
    are provided separately to processing methods.

    Attributes:
        output_directory: Directory where results will be written
        pixel_size: Optional pixel size in meters. Ignored if image resolution
                   can be read from input images
        title_prefix: Optional prefix for chart titles (max 26 characters)
        gen_kp_mask: Whether to generate a TIFF mask based on key points
        gen_delta_raster: Whether to generate intermediate products (dx/dy raster)
        generate_kp_chips: Whether to generate KP chip images
        dem_description: Optional DEM source description for plots
        enable_large_shift_detection: Whether to detect and correct large pixel shifts
    """

    output_directory: Path
    pixel_size: Optional[float]
    title_prefix: Optional[str]
    gen_kp_mask: bool
    gen_delta_raster: bool
    generate_kp_chips: bool
    dem_description: Optional[str]
    enable_large_shift_detection: bool
