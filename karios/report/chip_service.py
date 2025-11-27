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
"""Module to generate KP chip images of monitored and reference images."""

import glob
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal
from pandas import DataFrame, Series

from karios.core.image import GdalRasterImage, open_gdal_dataset

logger = logging.getLogger(__name__)


class CenterAndQuarterCellPointSelector:
    """
    Select points from image cells using center + quarter strategy.

    Strategy:
    1. Select point closest to cell center
    2. For each quarter, select point with radius closest to median radius of that quarter
    """

    def __init__(self, image_width, image_height, grid_size=(5, 5)):
        """
        Initialize the point selector.

        Args:
            image_width (int): Width of the image in pixels
            image_height (int): Height of the image in pixels
            grid_size (tuple): Grid dimensions (rows, cols). Default is (5, 5) for 25 cells
        """
        self.image_width = image_width
        self.image_height = image_height
        self.grid_rows, self.grid_cols = grid_size
        self.cell_width = image_width / self.grid_cols
        self.cell_height = image_height / self.grid_rows

    def select_points(self, df):
        """
        Select points using center + quarter strategy.

        Args:
            df (DataFrame): DataFrame with columns ['x0', 'y0', 'score'] at minimum

        Returns:
            DataFrame: DataFrame containing selected points
        """
        if df.empty:
            return pd.DataFrame()

        # Prepare dataframe with cell assignments
        df_work = self._assign_points_to_cells(df.copy())

        selected_points = []

        # Process each cell
        for cell_id in range(self.grid_rows * self.grid_cols):
            cell_points = df_work[df_work["cell_id"] == cell_id]

            if len(cell_points) == 0:
                continue

            # Process this cell
            cell_selected = self._process_cell(cell_points, cell_id)
            selected_points.extend(cell_selected)

        # Return final result
        return self._create_result_dataframe(selected_points)

    def _assign_points_to_cells(self, df):
        """Assign each point to its corresponding cell."""
        # Calculate which cell each point belongs to
        cell_col = np.floor(df["x0"] / self.cell_width).astype(int)
        cell_row = np.floor(df["y0"] / self.cell_height).astype(int)

        # Handle edge cases (points exactly on the right/bottom edge)
        cell_col = np.clip(cell_col, 0, self.grid_cols - 1)
        cell_row = np.clip(cell_row, 0, self.grid_rows - 1)

        # Create unique cell identifier
        df["cell_id"] = cell_row * self.grid_cols + cell_col

        return df

    def _process_cell(self, cell_points, cell_id):
        """Process a single cell to select center and quarter points."""
        selected = []

        # Calculate cell boundaries and center
        cell_bounds = self._calculate_cell_boundaries(cell_id)

        # Add distance to center for all points
        cell_points = self._add_center_distances(cell_points, cell_bounds)

        # Step 1: Select center point
        center_point = self._select_center_point(cell_points)
        selected.append(center_point)

        # Step 2: Select quarter points (excluding center point)
        remaining_points = cell_points[cell_points.index != center_point.name]

        if len(remaining_points) > 0:
            quarter_points = self._select_quarter_points(remaining_points, cell_bounds)
            selected.extend(quarter_points)

        return selected

    def _calculate_cell_boundaries(self, cell_id):
        """Calculate the boundaries and center of a cell."""
        cell_row_idx = cell_id // self.grid_cols
        cell_col_idx = cell_id % self.grid_cols

        # Calculate cell boundaries
        x_start = cell_col_idx * self.cell_width
        x_end = (
            (cell_col_idx + 1) * self.cell_width
            if cell_col_idx < self.grid_cols - 1
            else self.image_width
        )

        y_start = cell_row_idx * self.cell_height
        y_end = (
            (cell_row_idx + 1) * self.cell_height
            if cell_row_idx < self.grid_rows - 1
            else self.image_height
        )

        # Calculate center
        center_x = (x_start + x_end) / 2
        center_y = (y_start + y_end) / 2

        return {
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
            "center_x": center_x,
            "center_y": center_y,
        }

    def _add_center_distances(self, points, cell_bounds):
        """Add distance to cell center for all points."""
        points = points.copy()
        points["dist_to_center"] = np.sqrt(
            (points["x0"] - cell_bounds["center_x"]) ** 2
            + (points["y0"] - cell_bounds["center_y"]) ** 2
        )
        return points

    def _select_center_point(self, cell_points):
        """Select the point closest to the cell center."""
        # Find points with minimum distance to center
        min_distance = cell_points["dist_to_center"].min()
        center_candidates = cell_points[cell_points["dist_to_center"] == min_distance]

        # Tie-breaking: choose highest score
        if len(center_candidates) > 1:
            selected = center_candidates.loc[center_candidates["score"].idxmax()].copy()
        else:
            selected = center_candidates.iloc[0].copy()

        # Add metadata
        selected["selection_type"] = "center"
        selected["quarter_id"] = -1

        return selected

    def _select_quarter_points(self, remaining_points, cell_bounds):
        """Select points from each quarter based on median radius strategy."""
        quarter_points = []

        # Define quarter boundaries
        quarters = self._define_quarter_boundaries(cell_bounds)

        # Process each quarter
        for quarter_idx, quarter_bounds in enumerate(quarters):
            quarter_point = self._process_single_quarter(
                remaining_points, quarter_bounds, quarter_idx
            )
            if quarter_point is not None:
                quarter_points.append(quarter_point)

        return quarter_points

    def _define_quarter_boundaries(self, cell_bounds):
        """Define the boundaries for the 4 quarters within a cell."""
        quarter_width = (cell_bounds["x_end"] - cell_bounds["x_start"]) / 2
        quarter_height = (cell_bounds["y_end"] - cell_bounds["y_start"]) / 2

        x_start, x_end = cell_bounds["x_start"], cell_bounds["x_end"]
        y_start, y_end = cell_bounds["y_start"], cell_bounds["y_end"]
        x_mid = x_start + quarter_width
        y_mid = y_start + quarter_height

        return [
            # Top-left (quarter 0)
            {"x_min": x_start, "x_max": x_mid, "y_min": y_start, "y_max": y_mid},
            # Top-right (quarter 1)
            {"x_min": x_mid, "x_max": x_end, "y_min": y_start, "y_max": y_mid},
            # Bottom-left (quarter 2)
            {"x_min": x_start, "x_max": x_mid, "y_min": y_mid, "y_max": y_end},
            # Bottom-right (quarter 3)
            {"x_min": x_mid, "x_max": x_end, "y_min": y_mid, "y_max": y_end},
        ]

    def _process_single_quarter(self, points, quarter_bounds, quarter_idx):
        """Process a single quarter to select the median-radius point."""
        # Find points in this quarter
        quarter_points = self._get_quarter_points(points, quarter_bounds, quarter_idx)

        if len(quarter_points) == 0:
            return None

        # Calculate median radius and select closest point
        median_radius = quarter_points["dist_to_center"].median()
        quarter_points = quarter_points.copy()
        quarter_points["dist_to_median"] = np.abs(quarter_points["dist_to_center"] - median_radius)

        # Find point closest to median radius
        min_dist_to_median = quarter_points["dist_to_median"].min()
        median_candidates = quarter_points[quarter_points["dist_to_median"] == min_dist_to_median]

        # Tie-breaking: choose highest score
        if len(median_candidates) > 1:
            selected = median_candidates.loc[median_candidates["score"].idxmax()].copy()
        else:
            selected = median_candidates.iloc[0].copy()

        # Add metadata
        selected["selection_type"] = f"quarter_{quarter_idx}"
        selected["quarter_id"] = quarter_idx

        return selected

    def _get_quarter_points(self, points, quarter_bounds, quarter_idx):
        """Get all points within a specific quarter."""
        # Base mask for quarter boundaries
        mask = (
            (points["x0"] >= quarter_bounds["x_min"])
            & (points["x0"] < quarter_bounds["x_max"])
            & (points["y0"] >= quarter_bounds["y_min"])
            & (points["y0"] < quarter_bounds["y_max"])
        )

        # Handle edge case: points exactly on the right/bottom boundary
        # should be included in the rightmost/bottommost quarter
        if quarter_idx in [1, 3]:  # Right quarters
            mask |= points["x0"] == quarter_bounds["x_max"]
        if quarter_idx in [2, 3]:  # Bottom quarters
            mask |= points["y0"] == quarter_bounds["y_max"]

        return points[mask]

    def _create_result_dataframe(self, selected_points):
        """Create the final result dataframe from selected points."""
        if not selected_points:
            return pd.DataFrame()

        result_df = pd.DataFrame(selected_points)

        # Clean up temporary columns
        columns_to_drop = [
            "dist_to_median",
            "dist_to_center",
            "cell_id",
            "selection_type",
            "quarter_id",
        ]
        # for col_name in columns_to_drop:
        #     if col_name not in result_df.columns:
        #         columns_to_drop.remove(col_name)

        result_df = result_df.drop(columns=columns_to_drop, errors="ignore")
        return result_df.reset_index(drop=True)


class SimpleCellPointSelector:

    def __init__(self, image_width, image_height, grid_size=(5, 5)):
        """
        Select top N points with highest scores from image cells of a grid overlay on the image.

        Args:
            image_width (int): Width of the image in pixels
            image_height (int): Height of the image in pixels
            grid_size (tuple): Grid dimensions (rows, cols). Default is (5, 5) for 25 cells
        """
        self.image_width = image_width
        self.image_height = image_height
        self.grid_rows, self.grid_cols = grid_size
        self.cell_width = image_width / self.grid_cols
        self.cell_height = image_height / self.grid_rows

    def select_points(self, df, top_n=4) -> DataFrame:
        """
        Select top N points with highest scores from each cell of a grid overlay on the image.
        Handles non-divisible image dimensions by distributing remainder pixels to edge cells.

        Args:
            df (DataFrame): DataFrame with columns ['x0', 'y0', 'score'] at minimum
            image_width (int): Width of the image in pixels
            image_height (int): Height of the image in pixels
            grid_size (tuple): Grid dimensions (rows, cols). Default is (5, 5) for 25 cells
            top_n (int): Number of top-scoring points to select from each cell

        Returns:
            DataFrame: DataFrame containing the selected points

        Note:
        -----
        For non-divisible dimensions, floating-point cell boundaries are used.
        The rightmost and bottommost cells may be slightly larger to cover remainder pixels.
        Example: 9999x9998 image → cells are ~1999.8 x 1999.6 pixels each
        """

        # Make a copy to avoid modifying the original dataframe
        df_work = df.copy()

        # Assign each point to a cell
        # Calculate which cell each point belongs to
        cell_col = np.floor(df_work["x0"] / self.cell_width).astype(int)
        cell_row = np.floor(df_work["y0"] / self.cell_height).astype(int)

        # Handle edge cases (points exactly on the right/bottom edge)
        cell_col = np.clip(cell_col, 0, self.grid_cols - 1)
        cell_row = np.clip(cell_row, 0, self.grid_rows - 1)

        # Create a unique cell identifier
        df_work["cell_id"] = cell_row * self.grid_cols + cell_col

        # Select top N points from each cell based on score
        selected_points = []

        for cell_id in range(self.grid_rows * self.grid_cols):  # 0 to 24 for 5x5 grid
            cell_points = df_work[df_work["cell_id"] == cell_id]

            if len(cell_points) > 0:
                # Sort by score in descending order and take top N
                top_points = cell_points.nlargest(top_n, "score")
                selected_points.append(top_points)

        # Combine all selected points
        if selected_points:
            result_df = pd.concat(selected_points, ignore_index=True)
            # remove unwanted column
            result_df.drop("cell_id", axis=1, inplace=True)
            return result_df

        return DataFrame()


class ChipService:
    """
    This class provides a function that generate KP chip images.
    Chips size id 57x57 px.
    """

    def __init__(self):
        self._ouput_dir_name: str = "chips"
        self._chip_size = 57
        # to avoid recompute it for each KP
        self._chip_margin = int((self._chip_size - 1) / 2)

    def generate_chips(
        self,
        monitored: GdalRasterImage,
        reference: GdalRasterImage,
        points: DataFrame,
        confident_threshold: float,
        output_dir: str | Path,
    ):
        """
        This function generates a maximum of 100 chip images of KP for monitored and reference images.
        For reference image chips are centered based on KP coordinates (x0,y0).
        For monitored image chips are centered based on monitored KP cooredinates x = round(x0 + dx), y = round(y0 + dy)
        Selected KP have score gte to the confident threshold,
        and then the 4 top score by image grid (5x5) cell in order to have a better KP distribution.

        Args:
            monitored (GdalRasterImage): monitored image to extract chips from
            reference (GdalRasterImage): reference image to extract chips from
            points (DataFrame): dataframe with series x0, y0, dx, dy, score
            confident_threshold (float): threshold to select KP
            output_dir (str | Path): directory into which chips directory should be created
        """
        logger.info("Generate chips")
        # first filtering
        filtered_points = points[points["score"] >= confident_threshold]
        if len(filtered_points) == 0:
            logger.warning(
                "No KP found having score gte to confident threshold %s to extract chip",
                confident_threshold,
            )
            return

        # second filtering, 4 top score by image grid (5x5) cell to get a maximum of 100 KP
        # selecor = SimpleCellPointSelector(monitored.x_size, monitored.y_size)
        # remove unwanted column
        # filtered_points.drop("cell_id", axis=1, inplace=True)

        selecor = CenterAndQuarterCellPointSelector(monitored.x_size, monitored.y_size)

        filtered_points = selecor.select_points(filtered_points)

        logger.info(
            "Select %s/%s points (%.2f%%) based on confident threshold %s",
            len(filtered_points),
            points.size,
            100 * len(filtered_points) / len(points),
            confident_threshold,
        )

        # prepare out dirs by managing resume case
        chips_dir_path = Path(os.path.join(output_dir, self._ouput_dir_name))
        if os.path.exists(chips_dir_path):
            logger.warning("Chips output dir already exists, clean it")
            shutil.rmtree(chips_dir_path)

        os.mkdir(chips_dir_path)
        os.mkdir(chips_dir_path / monitored.file_name)
        os.mkdir(chips_dir_path / reference.file_name)

        # load only once mon and ref gdal dataset
        with open_gdal_dataset(monitored.filepath) as mon_dataset:
            with open_gdal_dataset(reference.filepath) as ref_dataset:
                # generate chips
                filtered_points.apply(
                    self._to_chips_gdal_dataset,
                    axis=1,
                    monitored=mon_dataset,
                    reference=ref_dataset,
                    out_dir=chips_dir_path,
                    monitored_filename=monitored.file_name,
                    reference_filename=reference.file_name,
                )

        logger.info("Chips generated in %s", chips_dir_path)

        filtered_points.to_csv(chips_dir_path / "chips.csv", sep=";", index=False)

        self._create_vrt(chips_dir_path / monitored.file_name, "monitored_chips.vrt")
        self._create_vrt(chips_dir_path / reference.file_name, "reference_chips.vrt")

    def _create_vrt(self, directory_path: Path, output_vrt_name="chips.vrt"):
        """
        Create a VRT file from all TIFF files in a directory.

        Args:
            directory_path (str): Path to directory containing TIFF files
            output_vrt_name (str): Name of output VRT file
        """

        # Change to the target directory
        original_dir = os.getcwd()
        os.chdir(directory_path)

        try:
            # Find all TIFF files in the directory
            tiff_files = glob.glob("*.TIFF")

            if not tiff_files:
                print(f"No TIFF files found in {directory_path}")
                return

            # Sort files for consistent ordering
            tiff_files.sort()

            # Create VRT file path (in same directory)
            vrt_path = directory_path / output_vrt_name

            # Build VRT options
            vrt_options = gdal.BuildVRTOptions(
                separate=False,
                srcNodata=None,
                VRTNodata=0,  # False for mosaic (single band)
            )

            # Create the VRT
            vrt_ds = gdal.BuildVRT(output_vrt_name, tiff_files, options=vrt_options)

            if vrt_ds is not None:
                # Close the dataset
                vrt_ds = None
                logger.info("Successfully created VRT: %s", vrt_path)

        except Exception as exc:
            logger.error("Unable to create VRT", exc_info=exc, stack_info=True)

        finally:
            # Return to original directory
            os.chdir(original_dir)

    def _to_chips_gdal_dataset(
        self,
        series: Series,
        monitored: gdal.Dataset,
        reference: gdal.Dataset,
        out_dir: Path,  # for out folder
        monitored_filename: str,  # for out folder
        reference_filename: str,  # for out folder
    ):
        """
        Generate KP chip using gdal translate for monitored and reference dataset in corresponding output dir.
        chips output dirs are out_dir/monitored_filename and out_dir/reference_filename

        Args:
            series (Series): _description_
            monitored (gdal.Dataset): _description_
            reference (gdal.Dataset): _description_
            out_dir (Path): _description_
        """
        # refrerence KP coordinates
        # x0 and y0 are always float with .O
        x0 = int(series["x0"])
        y0 = int(series["y0"])
        x0_offset = x0 - self._chip_margin
        y0_offset = y0 - self._chip_margin

        # monitored KP coordinates
        # dx and dy are float, use nearest int value
        x1 = round(series["x0"] + series["dx"])
        y1 = round(series["y0"] + series["dy"])
        x1_offset = x1 - self._chip_margin
        y1_offset = y1 - self._chip_margin

        if x0_offset < 0 or y0_offset < 0 or x1_offset < 0 or y1_offset < 0:
            # TODO: complete for right and bottom
            logger.warning("Chip to close to image boundaries, skip it")
            return

        ref_chip_path = out_dir / reference_filename / f"REF_{x0}_{y0}.TIFF"
        options = gdal.TranslateOptions(
            srcWin=[
                x0_offset,
                y0_offset,
                self._chip_size,
                self._chip_size,
            ],  # [xoff, yoff, xsize, ysize]
            format="GTiff",
        )
        dataset = gdal.Translate(ref_chip_path, reference, options=options)
        dataset.FlushCache()
        dataset = None

        mon_chip_path = out_dir / monitored_filename / f"MON_{x0}_{y0}.TIFF"
        options = gdal.TranslateOptions(
            srcWin=[
                x1_offset,
                y1_offset,
                self._chip_size,
                self._chip_size,
            ],  # [xoff, yoff, xsize, ysize]
            format="GTiff",
        )
        dataset = gdal.Translate(mon_chip_path, monitored, options=options)
        dataset.FlushCache()
        dataset = None
