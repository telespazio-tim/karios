# -*- coding: utf-8 -*-
# Copyright (c) 2025 Telespazio France.
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
"""Module to generate HTML reports for KARIOS results."""

import datetime
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from karios.api.config import RuntimeConfiguration
    from karios.api.core import AccuracyAnalysis, MatchResult, ReportPaths

logger = logging.getLogger(__name__)

CSS_STYLES = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f7f6;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 0;
            border-radius: 8px 8px 0 0;
            margin-bottom: 20px;
            overflow: hidden;
            position: relative;
        }
        .header-banner {
            width: 100%;
            display: block;
        }
        .header-content {
            padding: 20px;
            background: rgba(255, 255, 255, 0.8);
            position: absolute;
            bottom: 0;
            width: 100%;
        }
        h1, h2, h3 {
            margin-top: 0;
        }
        .section {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
        .image-container {
            text-align: center;
            margin-top: 20px;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        .image-title {
            font-weight: bold;
            margin-top: 10px;
            display: block;
        }
        .stats-card {
            background: #e9ecef;
            padding: 15px;
            border-radius: 6px;
            border-left: 5px solid #2c3e50;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #333;
            margin-top: 40px;
            padding: 80px 20px;
            border-top: 1px solid #ddd;
            background-image: url('footer_logo.svg');
            background-repeat: no-repeat;
            background-position: center;
            background-size: 200px;
            background-color: rgba(255, 255, 255, 0.85);
            background-blend-mode: lighten;
        }
        .links {
            margin-top: 10px;
        }
        .links a {
            color: #3498db;
            text-decoration: none;
            margin: 0 10px;
        }
        .links a:hover {
            text-decoration: underline;
        }
        .nav {
            display: flex;
            background: #2c3e50;
            padding: 10px 20px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .nav a {
            color: white;
            text-decoration: none;
            margin-right: 20px;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 4px;
        }
        .nav a.active {
            background: #34495e;
            color: #3498db;
        }
        .nav a:hover {
            background: #34495e;
        }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KARIOS Processing Report</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <header>
        <img src="karios_index_banner.webp" alt="KARIOS Banner" class="header-banner">
    </header>

    <nav class="nav">
        <a href="report.html" class="active">Summary</a>
        {products_link}
        {chips_link}
    </nav>

    <div class="section">
        <h1>KARIOS Processing Report</h1>
        <div class="grid">
            <div>
                <h3>Input Images</h3>
                <table>
                    <tr><th>Monitored</th><td>{monitored_image}</td></tr>
                    <tr><th>Reference</th><td>{reference_image}</td></tr>
                    <tr><th>Mask</th><td>{mask_file}</td></tr>
                    <tr><th>DEM</th><td>{dem_file}</td></tr>
                </table>
            </div>
            <div>
                <h3>Configuration</h3>
                <table>
                    <tr><th>Pixel Size</th><td>{pixel_size} m</td></tr>
                    <tr><th>Large Shift Detection</th><td>{large_shift_detection}</td></tr>
                    <tr><th>Title Prefix</th><td>{title_prefix}</td></tr>
                </table>
            </div>
        </div>
        <p>Generated on {generation_date}</p>
    </div>

    <div class="section">
        <h2>Accuracy Results</h2>
        <div class="grid">
            <div class="stats-card">
                <h3>Statistical Summary</h3>
                <table>
                    <tr><th>Matched Points</th><td>{matched_points}</td></tr>
                    <tr><th>Valid Pixels</th><td>{valid_pixels} / {total_pixels} ({valid_percent}%)</td></tr>
                    <tr><th>Mean X</th><td>{mean_x} pixels</td></tr>
                    <tr><th>Mean Y</th><td>{mean_y} pixels</td></tr>
                    <tr><th>Std Dev X</th><td>{std_x} pixels</td></tr>
                    <tr><th>Std Dev Y</th><td>{std_y} pixels</td></tr>
                    <tr><th>CE90</th><td>{ce90}</td></tr>
                    <tr><th>CE95</th><td>{ce95}</td></tr>
                </table>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Visualizations</h2>
        <div class="image-container">
            <span class="image-title">01 - Overview</span>
            <img src="{overview_plot}" alt="Overview Plot">
        </div>
        <div class="grid">
            <div class="image-container">
                <span class="image-title">02 - DX Shift</span>
                <img src="{dx_plot}" alt="DX Plot">
            </div>
            <div class="image-container">
                <span class="image-title">03 - DY Shift</span>
                <img src="{dy_plot}" alt="DY Plot">
            </div>
        </div>
        <div class="image-container">
            <span class="image-title">04 - Circular Error (CE)</span>
            <img src="{ce_plot}" alt="CE Plot">
        </div>
        {dem_plots_html}
    </div>

    <div class="footer">
        <p>KARIOS - KLT-based Algorithm for Registration of Images from Observing Systems</p>
        <div class="links">
            <a href="https://telespazio-tim.github.io/karios" target="_blank">Website</a> |
            <a href="https://github.com/telespazio-tim/karios" target="_blank">GitHub Repository</a>
        </div>
    </div>
</body>
</html>
"""

PRODUCTS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KARIOS Products - {title_prefix}</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <header>
        <img src="karios_index_banner.webp" alt="KARIOS Banner" class="header-banner">
    </header>

    <nav class="nav">
        <a href="report.html">Summary</a>
        <a href="products.html" class="active">Products</a>
        {chips_link}
    </nav>

    <div class="section">
        <h1>Optional Output Files</h1>
        <p>The following products were generated during processing:</p>
        <table>
            <thead>
                <tr>
                    <th>Product Type</th>
                    <th>File Name</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {products_rows}
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>KARIOS - KLT-based Algorithm for Registration of Images from Observing Systems</p>
        <div class="links">
            <a href="https://telespazio-tim.github.io/karios" target="_blank">Website</a> |
            <a href="https://github.com/telespazio-tim/karios" target="_blank">GitHub Repository</a>
        </div>
    </div>
</body>
</html>
"""

CHIPS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KARIOS Key Point Chips - {title_prefix}</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <header>
        <img src="karios_index_banner.webp" alt="KARIOS Banner" class="header-banner">
    </header>

    <nav class="nav">
        <a href="report.html">Summary</a>
        {products_link}
        <a href="chips.html" class="active">Chips</a>
    </nav>

    <div class="section">
        <h1>Key Point Chips</h1>
        <p>Visual verification chips for selected key points.</p>
        <div class="grid">
            <div class="stats-card">
                <h3>VRT Files</h3>
                <p>These files can be opened in QGIS to see all chips mosaicked:</p>
                <ul>
                    {chips_vrt_links}
                </ul>
            </div>
        </div>
        <div class="grid">
            <div class="stats-card">
                <h3>Chips CSV</h3>
                <p>Full list of selected key points used for chips: <a href="chips/chips.csv">chips.csv</a></p>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>KARIOS - KLT-based Algorithm for Registration of Images from Observing Systems</p>
        <div class="links">
            <a href="https://telespazio-tim.github.io/karios" target="_blank">Website</a> |
            <a href="https://github.com/telespazio-tim/karios" target="_blank">GitHub Repository</a>
        </div>
    </div>
</body>
</html>
"""


class HtmlReportGenerator:
    """Generator for HTML reports."""

    def __init__(
        self,
        output_dir: Path,
        match_result: "MatchResult",
        accuracy_analysis: "AccuracyAnalysis",
        report_paths: "ReportPaths",
        runtime_config: "RuntimeConfiguration",
        dem_file_path: Optional[Path] = None,
    ):
        self.output_dir = output_dir
        self.match_result = match_result
        self.accuracy_analysis = accuracy_analysis
        self.report_paths = report_paths
        self.runtime_config = runtime_config
        self.dem_file_path = dem_file_path

    def _copy_assets(self):
        """Copy required assets to output directory."""
        banner_src = Path(__file__).parent / "karios_index_banner.webp"
        if banner_src.exists():
            shutil.copy(banner_src, self.output_dir / "karios_index_banner.webp")
        else:
            logger.warning("Banner image not found at %s", banner_src)

        footer_logo_src = Path(__file__).parent / "footer_logo.svg"
        if footer_logo_src.exists():
            shutil.copy(footer_logo_src, self.output_dir / "footer_logo.svg")
        else:
            logger.warning("Footer logo image not found at %s", footer_logo_src)

    def generate(self) -> Path:
        """Generate the HTML report file(s)."""
        logger.info("Generating HTML report")

        self._copy_assets()

        # Check for products and chips to build navigation
        has_products = len(self.report_paths.products) > 0
        has_chips = self.runtime_config.generate_kp_chips

        products_link = '<a href="products.html">Products</a>' if has_products else ""
        chips_link = '<a href="chips.html">Chips</a>' if has_chips else ""

        dem_plots_html = ""
        if self.report_paths.dem_plots:
            dem_plots_html = "<h3>DEM-based Analysis</h3>"
            dem_plots_html += '<div class="grid">'
            for plot_path in self.report_paths.dem_plots:
                relative_path = Path(plot_path).name
                title = (
                    relative_path.replace("dem_", "").replace(".png", "").replace("_", " ").title()
                )
                dem_plots_html += f"""
                <div class="image-container">
                    <span class="image-title">DEM - {title}</span>
                    <img src="{relative_path}" alt="{title} Plot">
                </div>"""
            dem_plots_html += "</div>"

        # 1. Generate Summary Page (report.html)
        summary_content = HTML_TEMPLATE.format(
            css_styles=CSS_STYLES,
            products_link=products_link,
            chips_link=chips_link,
            generation_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            monitored_image=self.match_result.monitored_image.file_name,
            reference_image=self.match_result.reference_image.file_name,
            mask_file=self.match_result.mask.file_name if self.match_result.mask else "None",
            dem_file=self.dem_file_path.name if self.dem_file_path else "None",
            pixel_size=self.runtime_config.pixel_size if self.runtime_config.pixel_size else "Auto",
            large_shift_detection=(
                "Enabled" if self.runtime_config.enable_large_shift_detection else "Disabled"
            ),
            title_prefix=self.runtime_config.title_prefix or "None",
            matched_points=len(self.match_result.points),
            valid_pixels=self.accuracy_analysis.valid_pixels,
            total_pixels=self.accuracy_analysis.total_pixels,
            valid_percent=f"{(self.accuracy_analysis.valid_pixels / self.accuracy_analysis.total_pixels * 100):.2f}",
            mean_x=f"{self.accuracy_analysis.mean_x:.3f}",
            mean_y=f"{self.accuracy_analysis.mean_y:.3f}",
            std_x=f"{self.accuracy_analysis.std_x:.3f}",
            std_y=f"{self.accuracy_analysis.std_y:.3f}",
            ce90=f"{self.accuracy_analysis.ce90:.3f}",
            ce95=f"{self.accuracy_analysis.ce95:.3f}",
            overview_plot=Path(self.report_paths.overview_plot).name,
            dx_plot=Path(self.report_paths.dx_plot).name,
            dy_plot=Path(self.report_paths.dy_plot).name,
            ce_plot=Path(self.report_paths.ce_plot).name,
            dem_plots_html=dem_plots_html,
        )

        report_file = self.output_dir / "report.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(summary_content)

        # 2. Generate Products Page if needed
        if has_products:
            products_rows = ""
            for p in self.report_paths.products:
                p_path = Path(p)
                p_name = p_path.name
                p_type = "Vector (GeoJSON)" if p_name.endswith(".json") else "Raster (GeoTIFF)"
                if "mask" in p_name:
                    p_type = "Mask (GeoTIFF)"
                
                products_rows += f"""
                <tr>
                    <td>{p_type}</td>
                    <td>{p_name}</td>
                    <td><a href="{p_name}" download>Download</a></td>
                </tr>"""

            products_content = PRODUCTS_TEMPLATE.format(
                css_styles=CSS_STYLES,
                title_prefix=self.runtime_config.title_prefix or "KARIOS",
                chips_link=chips_link,
                products_rows=products_rows,
            )
            with open(self.output_dir / "products.html", "w", encoding="utf-8") as f:
                f.write(products_content)

        # 3. Generate Chips Page if needed
        if has_chips:
            mon_vrt = f"chips/{self.match_result.monitored_image.file_name}/monitored_chips.vrt"
            ref_vrt = f"chips/{self.match_result.reference_image.file_name}/reference_chips.vrt"
            
            chips_vrt_links = f'<li><a href="{mon_vrt}">Monitored Chips VRT</a></li>'
            chips_vrt_links += f'<li><a href="{ref_vrt}">Reference Chips VRT</a></li>'

            chips_content = CHIPS_TEMPLATE.format(
                css_styles=CSS_STYLES,
                title_prefix=self.runtime_config.title_prefix or "KARIOS",
                products_link=products_link,
                chips_vrt_links=chips_vrt_links,
            )
            with open(self.output_dir / "chips.html", "w", encoding="utf-8") as f:
                f.write(chips_content)

        logger.info("HTML report generated at %s", report_file)
        return report_file
