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


"""Program argument parser module."""
import argparse
import logging
import os
from argparse import ArgumentTypeError
from pathlib import Path, PurePath

LOGGER = logging.getLogger(__name__)


ROOT_DIR = Path(os.path.dirname(__file__))  # .parent.absolute()


def _validate_prefix(prefix) -> str:
    if len(prefix) <= 26:
        return prefix
    raise ArgumentTypeError("Title prefix is too long (>26 characters)")


def _validate_file(file_path):
    if not os.path.exists(file_path):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise ArgumentTypeError(f"{file_path} does not exist")

    if not os.path.isfile(file_path):
        raise ArgumentTypeError(f"{file_path} is not a file")

    return file_path


class KariosArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        """Init and configure KariosArgumentParser"""

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self._configure_arguments()
        self._args = None

    def parse_args(self, args=None, namespace=None):
        """Parse given arguments.

        Args:
        argv: list[str]: arg to parse

        Returns:
        Namespace: parsed args:
        - mon    Path to the monitored sensor product
        - ref    Path to the reference sensor product
        - mask   Path to the mask (default: None)
        - conf   Configuration file path
        - out    Output results folder path
        - resume Do not apply klt, just accuracy analysis (default: False)
        - pixel_size
        - gen_kp_mask
        - gen_delta_raster
        - title_prefix
        - dem_file_path
        - dem_description
        - no_log_file
        - debug
        - log_file_path

        """

        self._args = super().parse_args(args, namespace)
        return self._args

    def _configure_arguments(self):

        mandatory_args = self.add_argument_group("Mandatory arguments")
        mandatory_args.add_argument(
            metavar="MONITORED_IMAGE_PATH",
            dest="mon",
            help="Path to the monitored sensor product",
            type=_validate_file,
        )

        mandatory_args.add_argument(
            metavar="REFERENCE_IMAGE_PATH",
            dest="ref",
            help="Path to the reference sensor product",
            type=_validate_file,
        )

        # Processing argument section
        processing_args = self.add_argument_group("Processing options")

        processing_args.add_argument(
            "--conf",
            help="Configuration file path",
            default=PurePath(ROOT_DIR, "configuration/processing_configuration.json"),
            required=False,
        )

        processing_args.add_argument(
            "--resume",
            dest="resume",
            help="Do not run KLT matcher, only accuracy analysis and report generation",
            default=False,
            action="store_true",
            required=False,
        )

        processing_args.add_argument(
            "--mask",
            dest="mask_file_path",
            help="Path to the mask to apply to the reference image",
            required=False,
        )

        processing_args.add_argument(
            "--input-pixel-size",
            "-pxs",
            dest="pixel_size",
            type=float,
            # pylint: disable-next=line-too-long
            help="Input image pixel size in meter. Ignored if image resolution can be read from input image",
            default=None,
            required=False,
        )

        # output options
        outputs_args = self.add_argument_group("Output options")

        outputs_args.add_argument(
            "--out",
            help="Output results folder path",
            default=PurePath(ROOT_DIR.parent.absolute(), "results"),
            required=False,
        )

        outputs_args.add_argument(
            "--generate-key-points-mask",
            "-kpm",
            dest="gen_kp_mask",
            help="Generate a tiff mask based on KP from KTL",
            default=False,
            action="store_true",
            required=False,
        )

        outputs_args.add_argument(
            "--generate-intermediate-product",
            "-gip",
            dest="gen_delta_raster",
            help="Generate a two band tiff based on KP with band 1 dx and band 2 dy",
            default=False,
            action="store_true",
            required=False,
        )

        outputs_args.add_argument(
            "--title-prefix",
            "-tp",
            dest="title_prefix",
            help="Add prefix to title of generated output charts (limited to 26 characters)",
            type=_validate_prefix,
            default=None,
            required=False,
        )

        # DEM argument group section
        dem_args = self.add_argument_group("DEM arguments (optional)")

        dem_args.add_argument(
            "--dem-file-path",
            dest="dem_file_path",
            help='DEM file path. If given, "shift mean by altitude group plot" is generated.',
            required=False,
            type=_validate_file,
        )
        dem_args.add_argument(
            "--dem-description",
            dest="dem_description",
            help="""DEM source name.
            It is added in \"shift mean by altitude group plot\" DEM source (example: COPERNICUS DEM resample to 10m).
            Ignored if --dem-file-path is not given""",
            required=False,
        )

        # DEM argument group section
        experimental_features_args = self.add_argument_group("Experimental (optional)")

        experimental_features_args.add_argument(
            "--enable-large-shift-detection",
            dest="enable_large_shift_detection",
            help="""If enabled, KARIOS looks for large pixel shift between reference and monitored image.
            When a significant shift is detected, KARIOS shifts the monitored image according 
            to the offsets it computes and then process to the matching""",
            default=False,
            action="store_true",
        )

        # Logging argument group section
        logging_args = self.add_argument_group("Logging arguments (optional)")

        logging_args.add_argument(
            "--no-log-file",
            dest="no_log_file",
            action="store_true",
            help="Do not log in file",
        )
        logging_args.add_argument(
            "--debug",
            "-d",
            dest="debug",
            action="store_true",
            help="Enable Debug mode",
        )
        logging_args.add_argument(
            "--log-file-path",
            help="Log file path",
            default=PurePath(os.getcwd(), "karios.log"),
            required=False,
        )

    def verify_arguments(self):
        """checks arguments consistency"""

        if self._args.dem_file_path and not self._args.dem_description:
            LOGGER.warning(
                "DEM provided but not DEM description, consider to use --dem-description option"
            )

        if not self._args.dem_file_path and self._args.dem_description:
            LOGGER.warning(
                "DEM description provided but DEM file path, DEM description will be ignored"
            )
