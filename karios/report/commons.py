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
import os

import imageio.v2 as imageio

TPZ = imageio.imread(os.path.join(os.path.dirname(__file__), "TPZ_logo.png"))
EDAP = imageio.imread(os.path.join(os.path.dirname(__file__), "EDAP_logo.png"))
ESA = imageio.imread(os.path.join(os.path.dirname(__file__), "ESA_logo_2020_Deep.png"))


def add_logo(figure, logo_gd):
    ax_esa = figure.add_subplot(logo_gd[0, 0])
    ax_edap = figure.add_subplot(logo_gd[0, 1])
    ax_tpz = figure.add_subplot(logo_gd[0, 2])

    ax_esa.imshow(ESA)
    ax_edap.imshow(EDAP)
    ax_tpz.imshow(TPZ)

    ax_esa.axis("off")
    ax_edap.axis("off")
    ax_tpz.axis("off")
