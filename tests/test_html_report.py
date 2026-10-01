#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for inlining of the HTML report branding assets."""

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
    """An HtmlReportGenerator wired to mocks, producing all four pages."""
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
    report_paths.mosaic = "05_mosaic.avif"
    report_paths.overlay = "06_overlay.avif"

    runtime_config = MagicMock()
    runtime_config.pixel_size = 1.0
    runtime_config.enable_large_shift_detection = False
    runtime_config.title_prefix = "test"
    runtime_config.generate_kp_chips = True
    runtime_config.dem_description = "dem"
    runtime_config.output_directory = tmp_path

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

    pages = ["report.html", "mosaic.html", "overlay.html", "products.html", "chips.html"]
    for page in pages:
        content = (tmp_path / page).read_text(encoding="utf-8")
        assert BANNER_URI_PREFIX in content, f"{page} misses the inlined banner"
        assert LOGO_URI_PREFIX in content, f"{page} misses the inlined footer logo"


def test_generate_leaves_no_asset_files_behind(generator, tmp_path):
    """Pages are self-contained: the assets are no longer copied as sidecars."""
    generator.generate()

    assert not (tmp_path / _BANNER_ASSET).exists()
    assert not (tmp_path / _FOOTER_LOGO_ASSET).exists()

    for page in ["report.html", "mosaic.html", "overlay.html", "products.html", "chips.html"]:
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


def test_generate_orders_the_tabs(generator, tmp_path):
    """Every page shows Summary / Products / Mosaic / Overlay / Chips, its own tab active."""
    generator.generate()

    pages = ["report.html", "products.html", "mosaic.html", "overlay.html", "chips.html"]
    for page in pages:
        content = (tmp_path / page).read_text(encoding="utf-8")
        nav = content[content.index('<nav class="nav">') : content.index("</nav>")]
        assert [nav.index(f'href="{other}"') for other in pages] == sorted(
            nav.index(f'href="{other}"') for other in pages
        )
        assert nav.count('class="active"') == 1
        assert f'<a href="{page}" class="active">' in nav


def test_generate_mosaic_tab(generator, tmp_path):
    """The mosaic page shows the mosaic, linked to open it at native resolution, and every page links to it."""
    generator.generate()

    content = (tmp_path / "mosaic.html").read_text(encoding="utf-8")
    assert '<a href="05_mosaic.avif" target="_blank"><img src="05_mosaic.avif"' in content
    assert '<a href="mosaic.html" class="active">Mosaic</a>' in content
    assert "features broken at the tile edges" in content
    for page in ["report.html", "products.html", "overlay.html", "chips.html"]:
        assert '<a href="mosaic.html">Mosaic</a>' in (tmp_path / page).read_text(encoding="utf-8")


def test_generate_overlay_tab(generator, tmp_path):
    """The overlay page shows the overlay, linked to open it at native resolution, and every page links to it."""
    generator.generate()

    content = (tmp_path / "overlay.html").read_text(encoding="utf-8")
    assert "<h1>Overlay</h1>" in content
    assert '<a href="06_overlay.avif" target="_blank"><img src="06_overlay.avif"' in content
    assert '<a href="overlay.html" class="active">Overlay</a>' in content
    assert "red and cyan fringes along the features" in content
    for page in ["report.html", "products.html", "mosaic.html", "chips.html"]:
        assert '<a href="overlay.html">Overlay</a>' in (tmp_path / page).read_text(
            encoding="utf-8"
        )


def test_generate_without_mosaic(generator, tmp_path):
    """Without mosaic, neither the page nor the tab are generated."""
    generator.report_paths.mosaic = None

    generator.generate()

    assert not (tmp_path / "mosaic.html").exists()
    assert "mosaic.html" not in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_generate_without_overlay(generator, tmp_path):
    """Without overlay, neither the page nor the tab are generated."""
    generator.report_paths.overlay = None

    generator.generate()

    assert not (tmp_path / "overlay.html").exists()
    assert "overlay.html" not in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_user_values_are_escaped_in_every_page(generator, tmp_path):
    """Crafted file names, title prefix and products cannot inject markup into the pages."""
    tag = "<svg onload=alert(1)>"
    generator.match_result.monitored_image.file_name = f"{tag}.tif"
    generator.match_result.reference_image.file_name = 'ref" onmouseover="alert(2).tif'
    generator.runtime_config.title_prefix = "<script>alert(3)</script>"
    generator.report_paths.products = [f"{tag}.json"]

    generator.generate()

    for page in ["report.html", "mosaic.html", "overlay.html", "products.html", "chips.html"]:
        content = (tmp_path / page).read_text(encoding="utf-8")
        assert tag not in content, page
        assert '" onmouseover="' not in content, page
        assert "<script>alert(3)" not in content, page
    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "&lt;svg onload=alert(1)&gt;.tif" in report


def test_chip_links_are_percent_encoded(generator, tmp_path):
    """A file name with characters special to URLs still links to its chips directory."""
    generator.match_result.reference_image.file_name = "ref #1?.tif"

    generator.generate()

    chips = (tmp_path / "chips.html").read_text(encoding="utf-8")
    assert 'href="chips/ref%20%231%3F.tif/reference_chips.vrt"' in chips


def test_configuration_values_are_escaped(generator):
    """A configuration value carrying markup is shown as text."""
    generator.laplacian_polarity_label = "<img src=x onerror=alert(4)>"

    rows = generator._build_config_rows_html()

    assert "<img" not in rows
    assert "&lt;img src=x onerror=alert(4)&gt;" in rows


def test_chip_links_follow_directories_of_same_named_images(generator, tmp_path):
    """Same file names get suffixed chip directories: the report links to those."""
    generator.match_result.monitored_image.file_name = "B04.jp2"
    generator.match_result.reference_image.file_name = "B04.jp2"

    generator.generate()

    chips = (tmp_path / "chips.html").read_text(encoding="utf-8")
    assert 'href="chips/B04.jp2_monitored/monitored_chips.vrt"' in chips
    assert 'href="chips/B04.jp2_reference/reference_chips.vrt"' in chips
