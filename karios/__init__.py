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
"""KARIOS - KLT Algorithm for Registration of Images from Observing Systems.

A tool for comparing and matching images using KLT feature tracking.
"""

import matplotlib

# Every plot karios makes is written straight to a file (plt.savefig), never
# shown interactively, so a GUI backend is both unneeded and a source of
# platform fragility (e.g. a broken/missing Tcl-Tk install crashing figure
# creation on some CI runners). Set before any karios.report module gets a
# chance to import pyplot, since the backend can no longer change once a
# figure has been created.
matplotlib.use("Agg")

# pylint: disable=wrong-import-position
from karios.api.config import RuntimeConfiguration  # noqa: E402
from karios.api.core import KariosAPI  # noqa: E402
from karios.version import __version__  # noqa: E402
