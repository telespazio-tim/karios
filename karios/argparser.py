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
import os
from argparse import ArgumentTypeError, Namespace
from pathlib import Path, PurePath

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


def parse_args(argv: list[str]) -> Namespace:
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

    """
    # Parse the command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        dest="mon", help="Path to the monitored sensor product", type=_validate_file
    )
    parser.add_argument(
        dest="ref", help="Path to the reference sensor product", type=_validate_file
    )
    parser.add_argument("--mask", help="Path to the mask", required=False)
    parser.add_argument(
        "--conf",
        help="Configuration file path",
        default=PurePath(ROOT_DIR, "configuration/processing_configuration.json"),
        required=False,
    )
    parser.add_argument(
        "--out",
        help="Output results folder path",
        default=PurePath(ROOT_DIR.parent.absolute(), "results"),
        required=False,
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        help="Do not run KLT matcher, only accuracy analysis and report generation",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--generate-key-points-mask",
        "-kpm",
        dest="gen_kp_mask",
        help="Generate a tiff mask based on KP from KTL",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--generate-intermediate-product",
        "-gip",
        dest="gen_delta_raster",
        help="Generate a two band tiff based on KP with band 1 dx and band 2 dy",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--input-pixel-size",
        "-pxs",
        dest="pixel_size",
        type=float,
        # pylint: disable-next=line-too-long
        help="Input image pixel size in meter. Ignored if image resolution can be read from input image",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--title-prefix",
        "-tp",
        dest="title_prefix",
        help="Add prefix to title of generated output charts (limited to 26 characters)",
        type=_validate_prefix,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--no-log-file",
        dest="no_log_file",
        action="store_true",
        help="Do not log in file",
    )
    parser.add_argument(
        "--debug",
        "-d",
        dest="debug",
        action="store_true",
        help="Enable Debug mode",
    )
    parser.add_argument(
        "--log-file-path",
        help="Log file path",
        default=PurePath(os.getcwd(), "karios.log"),
        required=False,
    )
    return parser.parse_args(argv)
