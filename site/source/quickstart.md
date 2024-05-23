# Getting started

## Prerequisite

You need to install conda or miniconda.

We recommend [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).

## Installation

Create the `karios` conda environment

```bash
conda env create -y -f environment.yml
```

Then activate the environment: 

```bash
conda activate karios
```

> NOTICE: the conda environment stay active until you close your console session. You need to activate it every time you start a new session and want to use karios in the console session.

## Usage

Print help: 

```bash
python karios/karios.py --help
```

Run Karios: 

```bash
python karios/karios.py <Image to match> <Reference image>
```

### Simple example
```bash
python karios/karios.py \
    /data/12SYH/LS9_OLIL2F_20220824T175017_N0403_R035_T12SYH_20230214T164934.SAFE/GRANULE/L2F_T12SYH_A000000_20220824T175017_LS9_R035/IMG_DATA/L2F_T12SYH_20220824T175017_LS9_R035_B04_10m.TIF \
    /data/References/GRI/T12SYH_20220514T175909_B04.jp2
```


# Input data prerequisite and limitations

```{todo}
Define content
```

# Configuration

## Configuration file

The default configuration is located in [karios/configuration/processing_configuration.json](https://github.com/telespazio-tim/karios/tree/develop/karios/configuration/processing_configuration.json)

```{literalinclude} ../../karios/configuration/processing_configuration.json
:language: json
```

### klt_matching parameters


- `xStart` : image X margin to apply (margin is skipped by the matcher)
- `tile_size` : tile size to process by KTL in the input image
- `laplacian_kernel_size` : Aperture size used to compute the second-derivative filters of Laplacian process
-
The following parameter allows to control how to find the most prominent corners in the
reference image, as described by the OpenCV documentation goodFeaturesToTrack, after applying Laplacian.

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

### accuracy_analysis

- `confidence_threshold` : max score for points found by the matcher to use to compute statistics written in correl_res.txt.
If `None`, not applied.

### plot_configuration.overview

- `fig_size` : Size of the generated figure in inches
- `shift_colormap` : matplotlib color map name for the KP shift error scatter plot
- `shift_auto_axes_limit` : auto compute KP shift error colorbar scale
- `shift_axes_limit` : KP shift error colorbar maximum limit, N/A if `shift_auto_axes_limit` is `true`
- `theta_colormap` : matplotlib color map name for the KP theta error scatter plot

### plot_configuration.shift

- `fig_size` : Size of the generated figure in inches
- `scatter_colormap` : matplotlib color map name for the KP shift scatter plot
- `scatter_auto_limit` : auto compute KP shift scatter plot limit
- `scatter_min_limit` : KP shift scatter plot minimum limit, N/A if `scatter_auto_limit` is `true`
- `scatter_max_limit` : KP shift scatter plot maximum limit, N/A if `scatter_auto_limit` is `true`
- `histo_mean_bin_size` : KP shift histogram bin size (number of image row/col for the histogram bin)

### plot_configuration.ce
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