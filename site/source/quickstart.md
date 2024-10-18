# Getting started

## Installation

### Prerequisite

You need to install conda or miniconda.

We recommend [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).

### Create Python environment 

Create the `karios` conda environment

```bash
conda env create -y -f environment.yml
```

Then activate the environment: 

```bash
conda activate karios
```

You are ready to run KARIOS.

:::{note}
The conda environment stay active until you close your terminal session. You need to activate it every time you start a new session and want to use KARIOS in the terminal session.
:::

## Run KARIOS

### Usage

To print KARIOS CLI help and see options parameters run the following command: 

```bash
python karios/karios.py --help
```

For details, [see below](#cli-options)

KARIOS takes as mandatory inputs :
- monitored sensor image file
- reference sensor image file

Run KARIOS: 

```bash
python karios/karios.py <Image to match> <Reference image>
```

### Simple example

```bash
python karios/karios.py \
    /data/12SYH/LS9_OLIL2F_20220824T175017_N0403_R035_T12SYH_20230214T164934.SAFE/GRANULE/L2F_T12SYH_A000000_20220824T175017_LS9_R035/IMG_DATA/L2F_T12SYH_20220824T175017_LS9_R035_B04_10m.TIF \
    /data/References/GRI/T12SYH_20220514T175909_B04.jp2
```

With DEM:

```bash
python karios/karios.py \
    --dem-file-path /data/12SYH/dem10.tiff \
    --dem-description "Copernicus DSM_30 resample from 90m to 10m" \
    /data/12SYH/LS9_OLIL2F_20220824T175017_N0403_R035_T12SYH_20230214T164934.SAFE/GRANULE/L2F_T12SYH_A000000_20220824T175017_LS9_R035/IMG_DATA/L2F_T12SYH_20220824T175017_LS9_R035_B04_10m.TIF \
    /data/References/GRI/T12SYH_20220514T175909_B04.jp2
```

### Input data prerequisite and limitations

Input image files shall contain only one layer of data, and the format shall recognized by gdal library.

Inputs images grids should be comparable. **The user should take care of its data preparation**.

That means geo coded images must have the same footprint, same geo transform information (same EPSG code) and same resolution. Image pixel resolution should also be square (same X,Y) and unit meter.

These constraints are also applicable to the DEM file provided with the option `--dem-file-path`.


### CLI Options

```
usage: karios.py [-h] [--conf CONF] [--resume] [--mask MASK_FILE_PATH] [--input-pixel-size PIXEL_SIZE] [--out OUT] [--generate-key-points-mask] [--generate-intermediate-product] [--title-prefix TITLE_PREFIX]
                 [--dem-file-path DEM_FILE_PATH] [--dem-description DEM_DESCRIPTION] [--enable-large-shift-detection] [--no-log-file] [--debug] [--log-file-path LOG_FILE_PATH]
                 MONITORED_IMAGE_PATH REFERENCE_IMAGE_PATH

options:
  -h, --help            show this help message and exit

Mandatory arguments:
  MONITORED_IMAGE_PATH  Path to the monitored sensor product
  REFERENCE_IMAGE_PATH  Path to the reference sensor product

Processing options:
  --conf CONF           Configuration file path (default: /opt/karios/karios/configuration/processing_configuration.json)
  --resume              Do not run KLT matcher, only accuracy analysis and report generation (default: False)
  --mask MASK_FILE_PATH
                        Path to the mask to apply to the reference image (default: None)
  --input-pixel-size PIXEL_SIZE, -pxs PIXEL_SIZE
                        Input image pixel size in meter. Ignored if image resolution can be read from input image (default: None)

Output options:
  --out OUT             Output results folder path (default: /opt/karios/results)
  --generate-key-points-mask, -kpm
                        Generate a tiff mask based on KP from KTL (default: False)
  --generate-intermediate-product, -gip
                        Generate a two band tiff based on KP with band 1 dx and band 2 dy (default: False)
  --title-prefix TITLE_PREFIX, -tp TITLE_PREFIX
                        Add prefix to title of generated output charts (limited to 26 characters) (default: None)

DEM arguments (optional):
  --dem-file-path DEM_FILE_PATH
                        DEM file path. If given, "shift mean by altitude group plot" is generated. (default: None)
  --dem-description DEM_DESCRIPTION
                        DEM source name. It is added in "shift mean by altitude group plot" DEM source (example: COPERNICUS DEM resample to 10m). Ignored if --dem-file-path is not given (default: None)

Experimental (optional):
  --enable-large-shift-detection
                        If enabled, KARIOS looks for large pixel shift between reference and monitored image. When a significant shift is detected, KARIOS shifts the monitored image according to the offsets it computes
                        and then process to the matching (default: False)

Logging arguments (optional):
  --no-log-file         Do not log in file (default: False)
  --debug, -d           Enable Debug mode (default: False)
  --log-file-path LOG_FILE_PATH
                        Log file path (default: /opt/karios/karios.log)
```

# Configuration

## Configuration file

The default configuration is located in [karios/configuration/processing_configuration.json](https://github.com/telespazio-tim/karios/tree/develop/karios/configuration/processing_configuration.json)

```{literalinclude} ../../karios/configuration/processing_configuration.json
:language: json
```

- `processing_configuration.shift_image_processing` parameters (Large Shift Matching processing parameters)
  - `bias_correction_min_threshold`: number of pixel threshold from which large shift is applied.

- `processing_configuration.klt_matching` parameters (Matching processing parameters)
  - `xStart` : image X margin to apply (margin is skipped by the matcher)
  - `tile_size` : tile size to process by KTL in the input image
  - `laplacian_kernel_size` : Aperture size used to compute the second-derivative filters of Laplacian process  
> The next parameters allow to control how to find the most prominent corners in the reference image, as described by the OpenCV documentation `goodFeaturesToTrack`, after applying the Laplacian operator.  
-
  - `minDistance` : Minimum possible Euclidean distance between the returned corners.
  - `blocksize` : Size of an average block for computing a derivative covariation matrix over each pixel neighbourhood.
  - `maxCorners` : Maximum number of corners to extract. If there are more corners than are found, the strongest of them is returned.
  `maxCorners = 0` implies that no limit on the maximum is set and all detected corners are returned.
  - `qualityLevel` : Parameter characterizing the minimal accepted quality of image corners.
  The parameter value is multiplied by the best corner quality measure.
  The corners with the quality measure less than the product are rejected.
  For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01,
  then all the corners with the quality measure less than 15 are rejected.
  - `matching_winsize` : size of the search window during matching corners in the reference and the monitored Laplacian images.
  - `outliers_filtering` : whether to filter or not the outliers points found during the matching.

Refer to section [KLT param leverage](#klt-param-leverage) for details

- `processing_configuration.accuracy_analysis`
  - `confidence_threshold` : max score for points found by the matcher to use to compute statistics written in correl_res.txt.
  If `None`, not applied.

- `plot_configuration.overview` (overview plot parameters)
  - `fig_size` : Size of the generated figure in inches
  - `shift_colormap` : matplotlib color map name for the KP shift error scatter plot
  - `shift_auto_axes_limit` : auto compute KP shift error colorbar scale
  - `shift_axes_limit` : KP shift error colorbar maximum limit, N/A if `shift_auto_axes_limit` is `true`
  - `theta_colormap` : matplotlib color map name for the KP theta error scatter plot

- `plot_configuration.shift` (shift by row/col group plot parameters)
  - `fig_size` : Size of the generated figure in inches
  - `scatter_colormap` : matplotlib color map name for the KP shift scatter plot
  - `scatter_auto_limit` : auto compute KP shift scatter plot limit
  - `scatter_min_limit` : KP shift scatter plot minimum limit, N/A if `scatter_auto_limit` is `true`
  - `scatter_max_limit` : KP shift scatter plot maximum limit, N/A if `scatter_auto_limit` is `true`
  - `histo_mean_bin_size` : KP shift histogram bin size (number of image row/col for the histogram bin)

- `plot_configuration.dem` (shift by altitude group plot parameters)
  - `fig_size` : Size of the generated figure in inches
  - `show_fliers` : draw fliers of box plot
  - `histo_mean_bin_size`: KP altitude histogram bin size (altitude ranges size)

- `plot_configuration.ce` (Circular error plot parameters)
  - `fig_size` : Height size of the generated figure in inches, width is 5/3 of the height
  - `ce_scatter_colormap` : matplotlib color map name for the KP shift density scatter plot

## KLT param leverage

### maxCorners & tile_size

In order to have a lower memory usage during KLT process, it is possible to define a tile size to process for KLT.

For example, a tile_size of 10000 for an image having a size of 20000x20000 pixels will result of 4 tiles to process.

In this context, the KLT process will look in each tiles for `maxCorners`.

While an image of 20000x20000 pixels results of 4 equals tiles, an image of 20000x15000 pixels also result of 4 tiles, but with different size, two of 10000x10000 pixels and two of 10000x5000 pixels.

The consequence is that the density for matching point will not be the same each tiles, the bigger tiles will have a lower matching point than the smallest.

You may also consider that the image can content empty parts where KLT will not find any matching point. So tiles having a large empty parts will also results to a bigger matching point density.

In order to avoid density difference in the final result, you can define a `tile_size` largest than the image with an hight `maxCorners`, or a small `tile_size` and `maxCorners` in order to have tiles with almost same size.

For example, for image of 20000x15000 pixels, you should consider a `tile_size` of 20000 (1 tile), or 5000 (12 equal tiles)