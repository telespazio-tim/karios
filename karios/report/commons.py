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
from abc import ABC, abstractmethod
from pathlib import Path

import imageio.v2 as imageio
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec

TPZ = imageio.imread(os.path.join(os.path.dirname(__file__), "TPZ_logo.png"))
EDAP = imageio.imread(os.path.join(os.path.dirname(__file__), "EDAP_logo.png"))
ESA = imageio.imread(os.path.join(os.path.dirname(__file__), "ESA_logo_2020_Deep.png"))


# TODO: add to AbstractPlot class and call in AbstractPlot.plot()
def add_logo(figure: Figure, logo_gd: GridSpecFromSubplotSpec):
    ax_esa = figure.add_subplot(logo_gd[0, 0])
    ax_edap = figure.add_subplot(logo_gd[0, 1])
    ax_tpz = figure.add_subplot(logo_gd[0, 2])

    ax_esa.imshow(ESA)
    ax_edap.imshow(EDAP)
    ax_tpz.imshow(TPZ)

    ax_esa.axis("off")
    ax_edap.axis("off")
    ax_tpz.axis("off")


class AbstractPlot(ABC):
    """Abstract class for plot reports classes
    Concrete implementations should:
    - provide the base plot/figure title that to their `_figure_title` implementation.
    Figure title is then completed by the abstraction with prefix if any.
    The abstraction set the figure title with it.
    - prepare the `matplotlib.figure.Figure` thanks to their implementation of `_prepare_figure`.
    - set the content of the `matplotlib.figure.Figure` with then implementation of the `_plot` method.
    The abstraction set the figure title with it.
    The abstraction also have in charge to close matplotlib plt and save the figure into file.
    """

    def __init__(self, title_prefix: str | None, fig_size):
        self._title_prefix = title_prefix
        self._figure = self._prepare_figure(fig_size)

    @property
    @abstractmethod
    def _figure_title(self) -> str: ...

    @abstractmethod
    def _plot(self): ...

    @abstractmethod
    def _prepare_figure(self, fig_size) -> Figure: ...

    def _get_title(self) -> str:
        if self._title_prefix:
            return f"{self._title_prefix}: {self._figure_title}"
        return self._figure_title

    def plot(self, output_file: Path):
        """Plot and save in output_file

        Args:
            output_file (Path): destination file path
        """
        self._plot()
        self._figure.suptitle(
            self._get_title(),
            size="16",
            ha="center",
        )

        plt.savefig(output_file)
        plt.close()
