# -*- coding: utf-8 -*-
# Copyright (c) 2026 Telespazio France.
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
import os
import re


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other security issues.
    """
    if not filename:
        return "unnamed"

    while re.search(r"\.\.[/\\]", filename):
        filename = re.sub(r"\.\.[/\\]", "", filename)

    filename = filename.strip()
    filename = re.sub(r"[^\w\.\-]", "_", filename)

    if not filename or filename.strip("_") == "":
        return "unnamed"

    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    if os.path.splitext(filename)[0].upper() in reserved_names:
        filename = f"safe_{filename}"

    return filename


def get_filename(path: str) -> str:
    """Extract path filename without extension and sanitize it."""
    filename = os.path.splitext(os.path.basename(path))[0]
    return sanitize_filename(filename)
