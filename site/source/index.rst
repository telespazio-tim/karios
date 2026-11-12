.. Karios documentation master file, created by
   sphinx-quickstart on Thu May 16 15:01:02 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. disable right sidebar

:html_theme.sidebar_secondary.remove:

.. raw:: html

    <style type="text/css">
         h1 {display: none;}
        .bd-main .bd-content .bd-article-container {max-width: 100%;}
        .big-font {font-size: var(--pst-font-size-h4); } /*color: var(--pst-color-primary)}*/
        .bd-header-article{display: none;} /* hide breadcrumbs*/
    </style>

.. .. important::
..     Site under construction

.. This title is not displayed due to custom css above (h1)

Welcome to Karios's documentation!
==================================

.. image:: _static/_images/karios_index_banner.png
    :align: center
    :class: light-dark

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   quickstart
   usage
   case_study/index
   cookbook/index
   references

.. raw:: html
   
    <br>

.. include:: welcome.md
   :parser: myst_parser.sphinx_

Videos
======

.. raw:: html

    <h3>Understanding the KARIOS report</h3>
    <p>Tour of the HTML report: summary, products, mosaic, overlay and Laplacian chips.</p>
    <video controls preload="none" width="100%" style="max-width: 960px;">
      <source src="_static/videos/karios-report-demo.mp4" type="video/mp4">
    </video>

.. raw:: html

    <h3>Scenario 1 - Landsat 8 / Sen2Like</h3>
    <p>Dense matching (minDistance 5) of a Sen2Like L2H image against Landsat 8.</p>
    <video controls preload="none" width="100%" style="max-width: 960px;">
      <source src="_static/videos/scenario-1-ls8-sen2like.mp4" type="video/mp4">
    </video>

.. raw:: html

    <h3>Scenario 2 - Sentinel-2A / Sentinel-2C commissioning</h3>
    <p>Key points mask output, explored in QGIS down to the metadata of one key point.</p>
    <video controls preload="none" width="100%" style="max-width: 960px;">
      <source src="_static/videos/scenario-2-s2a-s2c-commissioning.mp4" type="video/mp4">
    </video>

.. raw:: html

    <h3>Scenario 3 - Landsat MSS / Sentinel-2 GRI with a DEM</h3>
    <p>Shift analysis against altitude using a DEM.</p>
    <video controls preload="none" width="100%" style="max-width: 960px;">
      <source src="_static/videos/scenario-3-mss-gri-dem.mp4" type="video/mp4">
    </video>

.. raw:: html

    <h3>Scenario 4 - SPOT / BSG large offset</h3>
    <p>Detect a large offset, correct it with karios align, then process again.</p>
    <video controls preload="none" width="100%" style="max-width: 960px;">
      <source src="_static/videos/scenario-4-spot-bsg-large-offset-align.mp4" type="video/mp4">
    </video>

.. raw:: html

    <h3>Scenario 5 - SAR / Sentinel-2</h3>
    <p>Multi-sensor matching, SAR against optical.</p>
    <video controls preload="none" width="100%" style="max-width: 960px;">
      <source src="_static/videos/scenario-5-sar-sentinel2.mp4" type="video/mp4">
    </video>

.. raw:: html

    <h3>Scenario 6 - Interband and key point chips</h3>
    <p>Registration between two bands, with key point chips (-gkc).</p>
    <video controls preload="none" width="100%" style="max-width: 960px;">
      <source src="_static/videos/scenario-6-interband.mp4" type="video/mp4">
    </video>

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
