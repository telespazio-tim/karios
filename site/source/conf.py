# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime

project = "KARIOS"
copyright = f"2023-{datetime.now().year}, Telespazio"
author = "telespazio-tim"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinxcontrib.images",
]

templates_path = ["_templates"]
exclude_patterns = []
todo_include_todos = True

myst_enable_extensions = ["fieldlist", "colon_fence"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_context = {"default_mode": "light"}  # Force Light mode

# specify the location of your github repo
html_theme_options = {
    "github_url": "https://github.com/telespazio-tim/karios/",
    "logo": {
        "text": "KARIOS",
        # "image_light": "_static/logo-light.png",
        # "image_dark": "_static/logo-dark.png",
    },
    "navbar_end": [
        "navbar-icon-links"
    ],  # Do not show theme switcher button: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/light-dark.html#light-and-dark-themes
    "show_toc_level": 2,
}

html_logo = "_static/logo.png"

html_css_files = [
    "css/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
]

# Work around for : https://github.com/pydata/pydata-sphinx-theme/issues/1662
html_sidebars = {
    "quickstart": [],
    # "cookbook/index": [],
    # "cookbook/config": [],
    # "cookbook/output": [],
    "references": [],
}

html_show_sourcelink = False
