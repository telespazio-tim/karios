#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for inlining of the HTML report branding assets."""

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from karios.core.configuration import (
    AccuracyAnalysisConfiguration,
    KLTConfiguration,
    ProcessingConfiguration,
    ShiftConfiguration,
)
from karios.report import html_report
from karios.report.html_report import (
    _BANNER_ASSET,
    _FOOTER_LOGO_ASSET,
    _FOOTER_LOGO_URI_TOKEN,
    HtmlReportGenerator,
    _asset_data_uri,
    _build_css_styles,
    _build_header_banner_html,
)

BANNER_URI_PREFIX = "data:image/webp;base64,"
LOGO_URI_PREFIX = "data:image/svg+xml;base64,"


def test_asset_data_uri_encodes_packaged_asset():
    """Data URI carries the asset MIME type and its verbatim bytes."""
    uri = _asset_data_uri(_BANNER_ASSET)

    assert uri.startswith(BANNER_URI_PREFIX)
    expected = (Path(html_report.__file__).parent / _BANNER_ASSET).read_bytes()
    assert base64.b64decode(uri[len(BANNER_URI_PREFIX) :]) == expected


def test_asset_data_uri_missing_asset_returns_none(caplog):
    """A missing asset is reported but does not raise."""
    assert _asset_data_uri("does_not_exist.webp") is None
    assert "not available" in caplog.text


def test_asset_data_uri_unsupported_type_returns_none(caplog):
    """An asset with no known MIME type is skipped rather than mis-labelled."""
    assert _asset_data_uri("notes.txt") is None
    assert "unknown type" in caplog.text


def test_build_css_styles_inlines_footer_logo():
    """The footer watermark is a data URI, and no placeholder survives."""
    css = _build_css_styles()

    assert f"background-image: url('{LOGO_URI_PREFIX}" in css
    assert _FOOTER_LOGO_URI_TOKEN not in css
    assert _FOOTER_LOGO_ASSET not in css


def test_build_css_styles_drops_declaration_when_logo_missing(monkeypatch):
    """Without the logo, the declaration goes away instead of emitting url('')."""
    monkeypatch.setattr(html_report, "_asset_data_uri", lambda _: None)

    css = _build_css_styles()

    assert "background-image" not in css
    assert "url('')" not in css
    # The rest of the footer styling is untouched.
    assert "background-blend-mode: lighten;" in css


def test_build_header_banner_html_is_empty_when_banner_missing(monkeypatch):
    """Without the banner, no img tag is emitted rather than src=""."""
    monkeypatch.setattr(html_report, "_asset_data_uri", lambda _: None)

    assert _build_header_banner_html() == ""


@pytest.fixture(name="generator")
def generator_fixture(tmp_path):
    """An HtmlReportGenerator wired to mocks, producing all three pages."""
    match_result = MagicMock()
    match_result.monitored_image.file_name = "monitored.tif"
    match_result.reference_image.file_name = "reference.tif"
    match_result.mask = None
    match_result.points = [MagicMock()]

    accuracy_analysis = MagicMock()
    accuracy_analysis.valid_pixels = 100
    accuracy_analysis.total_pixels = 1000
    accuracy_analysis.mean_x = 0.1
    accuracy_analysis.mean_y = 0.2
    accuracy_analysis.std_x = 0.01
    accuracy_analysis.std_y = 0.02
    accuracy_analysis.ce90 = 0.5
    accuracy_analysis.ce95 = 0.6

    report_paths = MagicMock()
    report_paths.overview_plot = "overview.png"
    report_paths.dx_plot = "dx.png"
    report_paths.dy_plot = "dy.png"
    report_paths.ce_plot = "ce.png"
    report_paths.dem_plots = []
    report_paths.products = ["product.json"]

    runtime_config = MagicMock()
    runtime_config.pixel_size = 1.0
    runtime_config.enable_large_shift_detection = False
    runtime_config.title_prefix = "test"
    runtime_config.generate_kp_chips = True
    runtime_config.dem_description = "dem"
    runtime_config.output_directory = tmp_path
    runtime_config.laplacian_power = 1.0

    return HtmlReportGenerator(
        output_dir=tmp_path,
        match_result=match_result,
        accuracy_analysis=accuracy_analysis,
        report_paths=report_paths,
        runtime_config=runtime_config,
        dem_file_path=None,
    )


def test_generate_inlines_branding_in_every_page(generator, tmp_path):
    """All generated pages embed the banner and the footer logo."""
    generator.generate()

    pages = ["report.html", "products.html", "chips.html"]
    for page in pages:
        content = (tmp_path / page).read_text(encoding="utf-8")
        assert BANNER_URI_PREFIX in content, f"{page} misses the inlined banner"
        assert LOGO_URI_PREFIX in content, f"{page} misses the inlined footer logo"


def test_generate_leaves_no_asset_files_behind(generator, tmp_path):
    """Pages are self-contained: the assets are no longer copied as sidecars."""
    generator.generate()

    assert not (tmp_path / _BANNER_ASSET).exists()
    assert not (tmp_path / _FOOTER_LOGO_ASSET).exists()

    for page in ["report.html", "products.html", "chips.html"]:
        content = (tmp_path / page).read_text(encoding="utf-8")
        assert _BANNER_ASSET not in content
        assert _FOOTER_LOGO_ASSET not in content


def test_generate_succeeds_without_branding_assets(generator, tmp_path, monkeypatch):
    """A report is still produced when the assets are absent from the package."""
    monkeypatch.setattr(html_report, "_asset_data_uri", lambda _: None)

    generator.generate()

    content = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<header>" in content
    assert "KARIOS Processing Report" in content


def _klt_configuration(**overrides) -> KLTConfiguration:
    defaults = dict(
        minDistance=10,
        blocksize=15,
        maxCorners=20000,
        matching_winsize=25,
        qualityLevel=0.1,
        xStart=0,
        tile_size=20000,
        laplacian_kernel_size=7,
        outliers_filtering=False,
    )
    defaults.update(overrides)
    return KLTConfiguration(**defaults)


def test_config_rows_show_laplacian_power(generator):
    """The resolved Laplacian power used for the run must be visible."""
    generator.laplacian_ksize_label = "7"
    generator.laplacian_polarity_label = "Normal"
    generator.runtime_config.laplacian_power = 0.35

    html = generator._build_config_rows_html()

    assert "<tr><th>Laplacian Power</th><td>0.35</td></tr>" in html


def test_config_rows_do_not_duplicate_resolved_klt_fields(generator):
    """laplacian_kernel_size/laplacian_invert_polarity must appear only once
    (the resolved summary row), not again under their raw field names from
    the as-configured KLT Matching section - which, outside "auto" mode,
    would just repeat the exact same value under a different label."""
    generator.laplacian_ksize_label = "mon=5, ref=7 (auto)"
    generator.laplacian_polarity_label = "Normal"
    generator.processing_config = ProcessingConfiguration()
    generator.processing_config.klt_configuration = _klt_configuration(laplacian_kernel_size="auto")
    generator.processing_config.shift_image_processing_configuration = ShiftConfiguration(
        bias_correction_min_threshold=2
    )
    generator.processing_config.accuracy_analysis_configuration = AccuracyAnalysisConfiguration(
        confidence_threshold=0.4
    )

    html = generator._build_config_rows_html()

    assert html.count("laplacian_kernel_size") == 0
    assert html.count("laplacian_invert_polarity") == 0
    assert "<tr><th>Laplacian Kernel Size</th><td>mon=5, ref=7 (auto)</td></tr>" in html
    # Other KLT fields are still dumped as usual.
    assert "<tr><th>tile_size</th><td>20000</td></tr>" in html
