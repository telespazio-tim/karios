# Getting started

## Installation

### Prerequisites

KARIOS is a Python application for which you need a dedicated conda environnement.

- Python 3.12+
- [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). **We recommend Miniconda**.
- **libGL**: you need libGL for linux, depending on your distribution it can be libgl1-mesa-glx, mesa-libGL or another. Install it if you don't have it yet.

Get KARIOS:

Download KARIOS sources [from github](https://github.com/telespazio-tim/karios/releases)
or clone the KARIOS github repository: 

```shell
git clone https://github.com/telespazio-tim/karios.git
# go to karios dir
cd karios
```

### Environment Setup

1. **Setup the conda environment:**

```bash
conda env create -f environment.yml
```

If the following error message  `CondaValueError: prefix already exists: <PATH_TO_CONDA_DIR>/envs/karios` appears, that means you already have a `karios` conda environment.

**Update it** with following command:

```bash
conda env update -f environment.yml --prune
```

2. **Activate the environment:**

```bash
conda activate karios
```

3. **Install KARIOS**

```bash
pip install .
```

Karios is now installed in its conda envronement, you are ready to use it.

:::{note}
The conda environment stays active until you close your terminal session. You need to activate it every time you start a new session to use KARIOS.
:::

### Verify Installation

```bash
python -m karios --help
```

or

```bash
karios --help
```

## Usage

:::{important}
Always activate the environment before using the command-line
:::

```bash
conda activate karios
```

### Input Requirements and recommendations

KARIOS takes as inputs:
- **Monitored image** (mandatory): The image to analyze for shifts/changes
- **Reference image** (mandatory): The stable reference image for comparison
- **Mask file** (optional): Exclude pixels from matching (co-registered the monitored image, having byte values, 0 are excluded while 1 are considered has valid for process)
- **DEM file** (optional): Enable altitude-based analysis (compatible with reference image)

Requirements:

- Input images grids should be comparable. **The user should take care of data preparation**.  
  That means geo coded images must have the same footprint, same geotransform information (same cartographic projection, i.e. EPSG code) and same resolution.
- Image pixel resolution should also be square (same X, Y) and unit meter.

:::{note}
**This is also applicable** to DEM and mask files for compatibility requirements.
:::

Recommendation:

- The user shall carefully check the dynamic range of the monitored and reference images, because KARIOS converts these input data into integers.  
  For instance, providing float values between 0 and 1 will give very poor results. In that case, it is recommended to multiply the data by 100.
- Input files shall contain only one layer (band) of data, and the format shall be recognized by GDAL library.

### Breaking changes in 3.0.0

- **Refused inputs**: images and masks that read other data are refused. That covers network paths (`/vsicurl`, `/vsis3`, `http://`...), VRT and web-service rasters, and OGR VRT, WFS, database and GPSBabel vector masks. Convert them to a plain file (e.g. GeoTIFF) first.
- **Exit status**: `karios process` and `karios align` exit with status 1 on error (they always exited with 0). Scripts can now rely on it.
- **Output folder names**: names derived from the input files are sanitized. Path traversal sequences are removed, characters other than letters, digits, `_`, `.` and `-` become `_`, and Windows reserved names get a `safe_` prefix. Inputs with spaces or special characters get differently named output folders.
- **Pixel units**: images in a geographic CRS or on a rotated grid are measured in pixels, unless `--input-pixel-size` is given.
- **API**: `AccuracyAnalysis` has the new required fields `rmse`, `min_error`, `max_error`, `std_dev` and `num_points`. They are NaN (`num_points` is 0) when no key point is confident. Code that builds it by hand must pass them.


## Run KARIOS

Compare a monitored image to a reference image:

```bash
karios process monitored.tif reference.tif
```

Results are written to `./results`: an HTML report, the key points CSV, plots and statistics. Use `--out` to choose another folder.

Add an optional mask (use `-` to skip it) and a DEM for altitude analysis:

```bash
karios process monitored.tif reference.tif mask.tif dem.tif --out ./results
```

If the two images are not on the same grid (different size, resolution or footprint), or are rotated or distorted relative to each other, align the monitored image first:

```bash
karios align monitored.tif reference.tif --out ./aligned
```

`karios --help`, `karios process --help` and `karios align --help` list every option.

:::{seealso}
The [detailed usage](usage.md) page documents all options, the configuration file, the outputs and the advanced features (large shift detection, coarse-to-fine matching, Laplacian tuning, global alignment).
:::
