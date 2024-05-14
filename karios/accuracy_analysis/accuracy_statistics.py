# -*- coding: utf-8 -*-
# Copyright (c) 2024 Telespazio France.
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


"""Statistic module."""

import logging
import os
from dataclasses import dataclass

import numpy as np
from pandas import DataFrame

from core.configuration import AccuracyAnalysisConfiguration

logger = logging.getLogger()


class GeometricStat:
    """Class to compute statistic.
    TODO: describe computed stats
    """

    def __init__(
        self,
        config: AccuracyAnalysisConfiguration,
        points: DataFrame,
        carto=False,
        pixel_size=1,
    ):
        self.valid = False
        self.confidence = config.confidence_threshold
        self.total_pixel = ""  # Valid Pixel included no background
        self.sample_pixel = ""  # Sample of pixel used for statistics
        self.min_x = ""
        self.max_x = ""
        self.median_x = ""
        self.mean_x = ""
        self.std_x = ""
        self.min_y = ""
        self.max_y = ""
        self.median_y = ""
        self.mean_y = ""
        self.std_y = ""
        self.v_x = points["dx"]  # vector of dx displacements
        # reverse x (line/northing) if image have SRS (carto representation)
        if carto:
            self.v_x = -self.v_x
        self.v_y = points["dy"]  # vector of dy displacements
        self.v_c = points["score"]  # vector of dc displacements
        self.total_match = (self.v_x).shape[0]  # All pixels as input
        self.vx_th = np.array(self.v_x.shape).astype(float)
        self.vy_th = np.array(self.v_x.shape).astype(float)
        self.pixel_size = pixel_size
        self.apply_confidence(self.confidence)

    def apply_confidence(self, confidence_threshold):
        """Filter vx , vy, vc to keep only records with score above confidence threshold.

        Args:
          confidence_threshold:

        """
        vx = self.v_x
        vy = self.v_y
        vc = self.v_c
        self.confidence = confidence_threshold

        mas = vc.gt(confidence_threshold)
        logger.info(mas.value_counts())

        self.v_x_th = np.array((vx[mas]))
        self.v_y_th = np.array((vy[mas]))
        self.v_c_th = np.array((vc[mas]))

        self.sample_pixel = (self.v_x_th).shape[0]
        self.percentage_of_match = 100 * np.double(self.sample_pixel) / np.double(self.total_match)

        logger.info(
            "Size of Statistical Sample  : %s , by using confidence %s ",
            self.sample_pixel,
            self.confidence,
        )

        logger.info("Keep  %.2f%% of total match", self.percentage_of_match)

    def compute_stats(self, nb_pixels, confidence_threshold=None):
        """

        Args:
          nb_pixels:
          confidence_threshold:  (Default value = None)

        Returns:

        """
        if confidence_threshold is not None:
            self.apply_confidence(confidence_threshold)

        vx = self.v_x_th
        vy = self.v_y_th
        vc = self.v_c_th
        logger.info(" -- Compute Final Statistics :")
        self.total_pixel = nb_pixels
        self.sample_pixel = np.size(vx)
        self.percentage_of_pixel = 100 * np.double(self.sample_pixel) / np.double(self.total_pixel)

        if (np.size(vx)) > 0:
            self.valid = True
            self.min_x = np.min(vx)
            self.max_x = np.max(vx)
            self.median_x = np.median(vx)
            self.mean_x = np.mean(vx)
            self.std_x = np.std(vx)

            self.min_y = np.min(vy)
            self.max_y = np.max(vy)
            self.median_y = np.median(vy)
            self.mean_y = np.mean(vy)
            self.std_y = np.std(vy)

            self.min_c = np.min(vc)
            self.max_c = np.max(vc)
            self.median_c = np.median(vc)
            self.mean_c = np.mean(vc)
            self.std_c = np.std(vc)

            self.valid = True
        else:
            logger.warning("No data in DC above confidence threshold")
            self.valid = False

    def display_results(self):
        """Format and display statistics results."""
        logger.info("-- DX / DY  statistics : ")
        logger.info(
            " Direction         : total_valid_pixel sample_pixel confidence_th   min    max    median    mean    std "  # pylint: disable=line-too-long
        )
        chx = [
            str(self.total_pixel),
            str(self.sample_pixel),
            str(self.confidence),
            str(self.min_x),
            str(self.max_x),
            str(self.median_x),
            str(self.mean_x),
            str(self.std_x),
        ]
        chy = [
            str(self.total_pixel),
            str(self.sample_pixel),
            str(self.confidence),
            str(self.min_y),
            str(self.max_y),
            str(self.median_y),
            str(self.mean_y),
            str(self.std_y),
        ]

        logger.info("Accuracy analysis: \n")
        logger.info(" DX (line)        : %s", " ".join(chx))
        logger.info(" DY (px(column))  : %s", " ".join(chy))

    def get_string_block(self, scale_factor, direction="x"):
        """

        Args:
          scale_factor:
          direction:  (Default value = "x")

        Returns:

        """
        # Create a text block to be added in the figure of the plot
        if direction == "x":
            # Output string to be included into the text box
            ch0 = " ".join([f" Conf_Value : {self.confidence:.2f}"])
            ch1 = " ".join([f" %Conf Px   :{self.percentage_of_pixel:.2f}%"])
            ch2 = " ".join([f"Minimum    : {(self.min_x * scale_factor):.2f}"])
            ch3 = " ".join([f"Maximum   : {(self.max_x * scale_factor):.2f}"])
            ch4 = " ".join([f"Mean          : {(self.mean_x * scale_factor):.2f}"])
            ch5 = " ".join([f"Std Dev      : {(self.std_x * scale_factor):.2f}"])
            ch6 = " ".join([f"Median       : {(self.median_x * scale_factor):.2f}"])

        if direction == "y":
            # Output string to be included into the text box
            ch0 = " ".join([f" Conf_Value :{self.confidence:.2f}"])
            ch1 = " ".join([f" %Conf Px   :{self.percentage_of_pixel:.2f}%"])
            ch2 = " ".join([f"Minimum    : {(self.min_y * scale_factor):.2f}"])
            ch3 = " ".join([f"Maximum   : {(self.max_y * scale_factor):.2f}"])
            ch4 = " ".join([f"Mean          : {(self.mean_y * scale_factor):.2f}"])
            ch5 = " ".join([f"Std Dev      : {(self.std_y * scale_factor):.2f}"])
            ch6 = " ".join([f"Median       : {(self.median_y * scale_factor):.2f}"])

        ch = "\n ".join([ch0, ch1, ch2, ch3, ch4, ch5, ch6])

        return ch

    def compute_percentile(self, percent, factor):
        x = self.v_x_th * factor
        y = self.v_y_th * factor

        # Computation CE90 2D :
        v_s = np.sort(np.sqrt(x * x + y * y))

        p = percent * v_s.shape[0]
        perc = int(p)
        index_n1 = perc - 1
        index_n2 = index_n1 + 1

        perc_frac = p - perc
        ce = v_s[index_n1] + (v_s[index_n2] - v_s[index_n1]) * perc_frac
        return ce

    def update_statistic_file(self, ref: str, mon: str, out_file_path: str = None):
        """write stats in file

        Args:
            ref (str): reference image name
            mon (str): monitored image name
            out_file_path (str, optional): destination file.
                Defaults to None, in this case, destination file is "correl_res.txt" in current dir
        """
        # default name
        if out_file_path is None:
            out_file_path = os.path.join(os.getcwd(), "correl_res.txt")

        # add titles if first line to write
        if not os.path.exists(out_file_path):
            with open(out_file_path, "w", encoding="utf-8") as txt_file:
                titles = " ".join(
                    [
                        "refImg",
                        "secImg",
                        "total_valid_pixel",
                        "sample_pixel",
                        "confidence_th",
                        "min_x",
                        "max_x",
                        "median_x",
                        "mean_x",
                        "std_x",
                        "min_y",
                        "max_y",
                        "median_y",
                        "mean_y",
                        "std_y",
                    ]
                )
                txt_file.write(titles + "\n")

        # write line

        results = [
            str(self.total_pixel),
            str(self.sample_pixel),
            str(self.confidence),
            str(self.min_x),
            str(self.max_x),
            str(self.median_x),
            str(self.mean_x),
            str(self.std_x),
            str(self.min_y),
            str(self.max_y),
            str(self.median_y),
            str(self.mean_y),
            str(self.std_y),
        ]

        with open(out_file_path, "a", encoding="utf-8") as txt_file:
            txt_file.write(f"{ref} {mon} {' '.join(results)}\n")

        logger.info("-- Update text file  : %s", out_file_path)
