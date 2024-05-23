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
   cookbook/index
   case_study/index
   references

.. raw:: html
   
    <br>

.. include:: welcome.md
   :parser: myst_parser.sphinx_

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
