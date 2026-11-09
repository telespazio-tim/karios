#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the plot base class."""

from matplotlib import pyplot as plt

from karios.report.commons import AbstractPlot


class _TextPlot(AbstractPlot):
    """A plot showing `text`, as the report plots show file names."""

    def __init__(self, title_prefix, text):
        self._text = text
        super().__init__(title_prefix, 4)

    @property
    def _figure_title(self):
        return "Test"

    def _prepare_figure(self, fig_size):
        return plt.figure(figsize=(fig_size, fig_size))

    def _plot(self):
        self._figure.text(0.1, 0.5, self._text)


def test_dollar_signs_in_names_do_not_break_plots(tmp_path):
    """"$" pairs in a file name or title prefix are plain text, not mathtext.

    Read as mathtext, "a$^$b.tif" raised ValueError and aborted the report,
    and "band_$1$_$2$.tif" was drawn as a formula instead of its name.
    """
    output = tmp_path / "plot.png"

    _TextPlot("x$\\frac$y", "Monitored : a$^$b.tif").plot(output)

    assert output.stat().st_size > 0


def test_plain_text_does_not_leak_into_other_figures(tmp_path):
    """The setting only applies to KARIOS plots, not to the caller's own figures."""
    _TextPlot(None, "a $b").plot(tmp_path / "plot.png")

    assert plt.rcParams["text.parse_math"] is True
