# About website

Web site is build thanks to [**sphinx**](https://www.sphinx-doc.org/)

It uses : 
- [`pydata_sphinx_theme`](https://pydata-sphinx-theme.readthedocs.io/) for theme
- [`myst_parser`](https://myst-parser.readthedocs.io/) extension for markdown support
- [`sphinx_design`](https://sphinx-design.readthedocs.io/) for some stuff like grid and cards

Some extensions of myst are used : 
- `fieldlist` for sphinx metadata field in MD
- `colon_fence` for sphinx_design grid in MD

Custom CSS are located in `source/_static/css/custom.css`

Most of the pages are in MD, except main page (index) which is RST

There is a workaround for https://github.com/pydata/pydata-sphinx-theme/issues/1662 to hide left sidebar on some pages in `source/conf.py`. See attribute `html_sidebars` in [`source/conf.py`](source/conf.py).

## Build the site

```console
make html
```

The result is located in a folder named `build`

open build/index.html in a web browser to see the result