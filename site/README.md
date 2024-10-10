# About KARIOS website

Web site is build thanks to [**sphinx**](https://www.sphinx-doc.org/)

It uses : 
- [`pydata_sphinx_theme`](https://pydata-sphinx-theme.readthedocs.io/) for theme
- [`myst_parser`](https://myst-parser.readthedocs.io/) extension for markdown support
- [`sphinx_design`](https://sphinx-design.readthedocs.io/) for some stuff like grid and cards

Some extensions of myst are used : 
- `fieldlist` for sphinx metadata field in Markdown
- `colon_fence` for sphinx_design grid in Markdown

Custom CSS are located in `source/_static/css/custom.css`

Most of the pages are written in Markdown, except main page (index) which is RST

> There is a workaround for https://github.com/pydata/pydata-sphinx-theme/issues/1662 to hide left sidebar on some pages in `source/conf.py`. See attribute `html_sidebars` in [`source/conf.py`](source/conf.py).

## Build the site

### Pre requisite

Install sphinx and dependencies **in your karios conda environment**, so your KARIOS conda environment **MUST** be activated.

From the `site` folder run : 

```console
conda install -n karios -c conda-forge --file requirements.txt
```

### Build command

To build the site run the following command from `site` folder

```console
make html
```

The result is located in a folder named `build/html`

Open `build/html/index.html` in a web browser to see the result.

> **IMPORTANT**
> 
> :warning: Do not change `site/.gitignore`
> 
> :warning: Do not add/commit the `build` folder

## Deployment

The KARIOS website is automatically deployed to github pages : https://telespazio-tim.github.io/karios/

It is driven by the github workflow defined in [.github/workflows/site.yml](../.github/workflows/site.yml)

## Editing

TODO