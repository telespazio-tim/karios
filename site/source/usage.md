# Detailed usage

Full reference of the `karios` command line, configuration and outputs. For installation and a first run, see the [quickstart](quickstart.md).

:::{important}
**All commands below suppose that the karios conda environment is activated.**
:::

## CLI Usage

KARIOS has two commands: `karios process` measures the shifts between two images, `karios align` warps an image onto another one (see [Global alignment](#global-alignment-karios-align)).

### Command Structure

```bash
karios process MONITORED_IMAGE REFERENCE_IMAGE [MASK_FILE] [DEM_FILE] [OPTIONS]
```

### Arguments

- `MONITORED_IMAGE`: Path to the image to analyze for shifts/changes
- `REFERENCE_IMAGE`: Path to the stable reference image for comparison
- `MASK_FILE`: Optional mask file to exclude pixels from matching (use '-' to skip)
- `DEM_FILE`: Optional DEM file for altitude-based analysis

### Examples

#### Basic Processing

```bash
karios process monitored.tif reference.tif
```

#### With Mask

```bash
karios process monitored.tif reference.tif mask.tif
```

#### With Mask and DEM

```bash
karios process monitored.tif reference.tif mask.tif dem.tif \
  --dem-description "SRTM 30m resample to 10m"
```

#### DEM Only (No Mask)

```bash
karios process monitored.tif reference.tif - dem.tif \
  --dem-description "Copernicus DEM 30m"
```

#### Full Workflow with Options

```bash
karios process monitored.tif reference.tif mask.tif dem.tif \
  --out ./results \
  --generate-key-points-mask \
  --generate-intermediate-product \
  --title-prefix "MyAnalysis" \
  --dem-description "SRTM 30m" \
  --enable-large-shift-detection
```

#### Resume Previous Analysis

```bash
karios process monitored.tif reference.tif mask.tif dem.tif \
  --resume \
  --out ./existing_results
```

### CLI Options

#### Processing Options

| Option | Type | Description |
|--------|------|------------|
| `--conf` | FILE | Configuration file path. Default is the built-in configuration.<br>[default: <INSTALL_DIR>/karios/configuration/processing_configuration.json] |
| `--resume` | Flag | Do not run KLT matcher, only accuracy analysis and report generation. |
| `--input-pixel-size`, `-pxs` | FLOAT | Input image pixel size in meter. Ignored if image resolution can be read from input image. |
| `--vector-mask` | FILE | Vector mask (GeoJSON, Shapefile...) excluding pixels from matching. Rasterized on the monitored image grid. |
| `--no-value` | FLOAT | Filter out key points where the reference or monitored image has this DN value. Repeat it for several values (`--no-value 0 --no-value 255`). |

#### Output Options

| Option | Type | Description |
|--------|------|------------|
| `--out` | PATH | Output results folder path [default: results] |
| `--title-prefix`, `-tp` | TEXT | Add prefix to title of generated output charts.<br>(limited to 26 characters). |
| `--generate-key-points-mask`, `-kpm` | FLAG | Generate a tiff mask based on KP from KTL. |
| `--generate-intermediate-product`, `-gip` | FLAG | Generate a two-bands tiff based on KP with band 1 (dx) and band 2 (dy).|
| `--generate-kp-chips`, `-gkc` | FLAG | Generate chip images centered on key points of monitored and reference products. |
| `--mosaic-tile-size`, `-mts` | INTEGER | Generate a checkerboard mosaic of the monitored and reference images with tiles of this size in pixel (e.g. 128), shown in the HTML report. 0 disables it [default: 0] |
| `--generate-overlay`, `-gov` | FLAG | Generate an overlay of the monitored image, in red, and the reference image, in green and blue, shown in the HTML report. Aligned features are gray, shifted ones fringed in red and cyan |
| `--dem-description` | TEXT | DEM source name. Added in generated DEM plots.<br>Example: "COPERNICUS DEM resampled to 10m." |

#### Advanced Options

| Option | Type | Description |
|--------|------|------------|
| `--enable-large-shift-detection` | FLAG | Enable detection and correction of large pixel shifts. |
| `--enable-coarse-to-fine` | FLAG | Match by descending the image pyramid explicitly, keeping key points near data edges. Cannot be combined with `--enable-large-shift-detection` nor with `laplacian_kernel_size: "auto"`. See [Coarse-to-fine matching](#coarse-to-fine-matching). |
| `--laplacian-power` | FLOAT 0–1 | Shape of the Laplacian rescale feeding the KLT tracker: `0` keeps it linear, `1` saturates it to a near-binary response [default: 1.0]. See [Laplacian tuning](#laplacian-tuning). |

#### Logging Options

| Option | Type | Description |
|--------|------|------------|
| `--debug`, `-d` | FLAG | Enable Debug mode. |
| `--no-log-file` | FLAG | Do not log in file (not compatible with `--log-file-path`). |
| `--log-file-path` | PATH | Log file path [default: karios.log]. |

## Configuration

### Processing Configuration

The processing configuration defines algorithm parameters and is loaded from JSON files. The default configuration is located in [karios/configuration/processing_configuration.json](https://github.com/telespazio-tim/karios/tree/develop/karios/configuration/processing_configuration.json).

```{dropdown} Default configuration (click to open)

  ```{literalinclude} ../../karios/configuration/processing_configuration.json
  :language: json
  ```

This configuration includes:
- **KLT matching parameters**: Corner detection, window sizes, quality thresholds.
- **Accuracy analysis settings**: Confidence thresholds for statistical calculations.  
- **Plot configurations**: Figure sizes, color maps, axis limits.
- **Large shift detection**: Bias correction thresholds.

When using the CLI, runtime configuration is automatically created from command-line arguments.

### Parameter Details

- `processing_configuration.shift_image_processing` (Large Shift Matching processing parameters)
  - `bias_correction_min_threshold`: Pixel threshold for applying large shift correction.

- `processing_configuration.klt_matching` (Matching processing parameters)
  - `xStart`: X margin to skip during matching.
  - `tile_size`: Tile size for memory-efficient processing.
  - `laplacian_kernel_size`: Aperture size for Laplacian filtering: an integer, `{"mon": n, "ref": n}`, or `"auto"` to sweep the sizes (see [Laplacian tuning](#laplacian-tuning)).
  - `laplacian_invert_polarity`: Invert the monitored pixels before the Laplacian: `false` (default), `true` or `"auto"` to try both (see [Laplacian tuning](#laplacian-tuning)).
  - `minDistance`: Minimum distance between detected corners.
  - `blocksize`: Block size for derivative computation.
  - `maxCorners`: Maximum corners to extract per tile. `0` implies that no limit on the maximum is set and all detected corners are returned.
    > The following parameter allows to control how to find the most prominent corners in the
  reference image, as described by the OpenCV documentation goodFeaturesToTrack, after applying Laplacian.
  - `qualityLevel`: Minimum corner quality threshold.
  - `matching_winsize`: Search window size during matching.
  - `outliers_filtering`: Enable/disable outlier filtering.
  - `maxLevel`: Pyramid depth used when matching. Deeper pyramids capture larger displacements but, in the default matching mode, erode matching inward from every data edge.
    Matching mode itself is a runtime switch, not a configuration key: see the `--enable-coarse-to-fine` CLI option.

Refer to section [KLT param leverage](#klt-parameters-leverage) for details.

- `processing_configuration.accuracy_analysis`
  - `confidence_threshold`: Minimum confidence score for statistical analysis. If `None`, not applied.

Plot configuration parameters for `overview`, `shift`, `dem`, and `ce` plots control figure sizes, color maps, and axis limits.

- `plot_configuration.overview` (overview plot parameters)
  - `fig_size` : size of the generated figure
  - `shift_colormap` : matplotlib color map name for the KP shift error scatter plot.
  - `shift_auto_axes_limit` : auto compute KP shift error colorbar scale.
  - `shift_axes_min_limit` : KP shift error colorbar minimum limit (default `0`), N/A if `shift_auto_axes_limit` is `true`.
  - `shift_axes_limit` : KP shift error colorbar maximum limit, N/A if `shift_auto_axes_limit` is `true`.
  - `theta_colormap` : matplotlib color map name for the KP theta error scatter plot.
  - `theta_min_limit`, `theta_max_limit` : KP angle error colorbar limits in degrees (default `-180`, `180`).

- `plot_configuration.shift` (shift by row/col group plot parameters)
  - `fig_size` : size of the generated figure.
  - `scatter_colormap` : matplotlib color map name for the KP shift scatter plot.
  - `scatter_auto_limit` : auto compute KP shift scatter plot limit.
  - `scatter_min_limit` : KP shift scatter plot minimum limit, N/A if `scatter_auto_limit` is `true`.
  - `scatter_max_limit` : KP shift scatter plot maximum limit, N/A if `scatter_auto_limit` is `true`.
  - `histo_mean_bin_size` : KP shift histogram bin size (number of image row/col for the histogram bin).

- `plot_configuration.dem` (shift by altitude group plot parameters)
  - `fig_size` : size of the generated figure.
  - `show_fliers` : draw fliers of box plot.
  - `histo_mean_bin_size`: KP altitude histogram bin size (altitude ranges size).

- `plot_configuration.ce` (Circular error plot parameters)
  - `fig_size` : height size of the generated figure, width is 5/3 of the height.
  - `ce_scatter_colormap` : matplotlib color map name for the KP shift density scatter plot.
  - `ce_auto_limit` : auto compute the CE plot axis limits from the largest displacement (default `true`).
  - `ce_axes_limit` : CE plot axis limit, the axes span `[-limit, limit]` (default `5.0`), N/A if `ce_auto_limit` is `true`.

### Outputs

KARIOS generates several types of outputs:

#### Statistical Files

- **CSV file**: Key points with dx/dy deviations and confidence scores (see [CSV Output section](#csv-output))
- **correl_res.txt**: Summary statistics (RMSE, min/max error, standard deviation, CE90, CE95, number of key points)

#### Visualizations

- **01_overview.png**: Error distribution overview with image thumbnails
- **02_dx.png**: X-direction displacement analysis by row/column
- **03_dy.png**: Y-direction displacement analysis by row/column  
- **04_ce.png**: Circular error analysis with statistical summaries
- **dem_*.png**: DEM-based altitude analysis (if DEM provided)
- **05_mosaic.avif** (lossless `.png` if Pillow has no AVIF support): Color checkerboard mosaic at native resolution, alternating reference (blue) and monitored (red) image tiles, each image histogram equalized to 8 bit, to eyeball misregistration at tile edges. Shown in the *Mosaic* tab of the HTML report (if `--mosaic-tile-size` is not 0)
- **06_overlay.avif** (lossless `.png` if Pillow has no AVIF support): Color overlay at native resolution, the monitored image in the red channel and the reference in the green and blue ones, each histogram equalized to 8 bit: aligned features are gray, shifted ones fringed in red and cyan. Shown in the *Overlay* tab of the HTML report (if `--generate-overlay`)

#### Products (Optional)

- **kp_mask.tif**: Binary mask of key point locations (if `--generate-key-points-mask`)
- **kp_delta.tif**: Two-band raster with dx/dy displacement values (if `--generate-intermediate-product`)
- **kp_delta.json**: GeoJSON of key points with displacement vectors (if images are georeferenced, see [GeoJSON Output section](#geojson-output))
- **chips/** directory: Directory with chip images of KP (see [Chip Images section](#chip-images-optional))

#### Configuration

- Copy of the processing configuration used.

## Output File Formats

### CSV Output

KARIOS generates a CSV file containing all key points detected by the KLT matcher with their displacement measurements and quality metrics. The CSV file is saved as `KLT_matcher_{monitored_filename}_{reference_filename}.csv` in the output directory.

#### CSV Columns

| Column | Description | Unit | Notes |
|--------|-------------|------|-------|
| `x0` | X coordinate of key point in reference image | pixels | Column position (0-based) |
| `y0` | Y coordinate of key point in reference image | pixels | Row position (0-based) |
| `dx` | Displacement in X direction (column) | pixels | Positive = eastward shift |
| `dy` | Displacement in Y direction (row) | pixels | Positive = southward shift |
| `score` | KLT matching confidence score | 0.0-1.0 | Higher values indicate better matches |
| `radial error` | Euclidean distance of displacement | pixels | √(dx² + dy²) |
| `angle` | Direction of displacement | degrees | Measured from east, counter-clockwise |
| `zncc_score` | Zero-mean Normalized Cross-Correlation score | -1.0 to 1.0 | Optional: only if large shift detection is disabled |

**Notes:**
- The CSV uses semicolon (`;`) as separator.
- `zncc_score` is only computed for a selection of key points with KLT score above the confidence threshold.
- `zncc_score` provides additional matching quality assessment using normalized cross-correlation.
- All coordinates are in image pixel space (not geographic coordinates).

### GeoJSON Output

When input images are georeferenced, KARIOS generates a GeoJSON file (`kp_delta.json`) containing key points as geographic features with displacement vectors and quality metrics.

#### GeoJSON Structure

```text
{
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {
      "name": "urn:ogc:def:crs:EPSG::{EPSG_CODE}"
    }
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [x_geographic, y_geographic]
      },
      "properties": {
        "dx": displacement_x_pixels,
        "dy": displacement_y_pixels,
        "score": klt_confidence_score,
        "radial error": euclidean_displacement,
        "angle": displacement_angle_degrees,
        "zncc_score": cross_correlation_score
      }
    }
  ]
}
```

#### Feature Properties

| Property | Description | Unit | Range |
|----------|-------------|------|-------|
| `dx` | X displacement in pixels | pixels | Real number |
| `dy` | Y displacement in pixels | pixels | Real number |
| `score` | KLT matching confidence | 0.0-1.0 | Higher = better match |
| `radial error` | Total displacement magnitude | pixels | ≥ 0 |
| `angle` | Displacement direction | degrees | -180 to 180 |
| `zncc_score` | Cross-correlation score | -1.0 to 1.0 | Optional, can be `null` |

**Coordinate System:**
- Point coordinates are in the same coordinate reference system as the input images.
- Coordinates are transformed from pixel space to geographic space using the image's geotransform.
- The CRS is specified in the `crs` object using the EPSG code from the reference image.

## Chip Images (Optional)

When called with `--generate-kp-chips`, KARIOS generates small image patches (chips) centered on key points for visual inspection and quality assessment.

```bash
karios process monitored.tif reference.tif --generate-kp-chips
```

### Chip Specifications

- **Size**: 57×57 pixels per chip
- **Selection**: Up to 125 chips using center + quarter strategy from a 5×5 grid overlay
- **Quality threshold**: Only key points with KLT score ≥ confidence threshold are considered
- **Pairing**: Each chip location generates two files (monitored and reference)

### Output Structure

```
results/
└── chips/
    ├── {monitored_filename}/
    │   ├── MON_1234_5678.TIFF
    │   ├── MON_2345_6789.TIFF
    │   └── ...
    ├── {reference_filename}/
    │   ├── REF_1234_5678.TIFF
    │   ├── REF_2345_6789.TIFF
    │   └── ...
    ├── chips.csv
    ├── monitored_chips.vrt
    └── reference_chips.vrt
```

### Generated Files

| File | Description |
|------|-------------|
| `MON_{x}_{y}.TIFF` | Chip from monitored image at reference coordinates (x,y) |
| `REF_{x}_{y}.TIFF` | Chip from reference image at coordinates (x,y) |
| `chips.csv` | Metadata of selected key points used for chip generation |
| `*_chips.vrt` | Virtual raster mosaics for easy visualization in GIS software |

:::{note}
- Chips near image boundaries are automatically excluded.
- Coordinate pairs (x,y) in filenames refer to the key point location in the reference image.
- Chips are extracted at the detected displacement location in the monitored image.
- VRT files enable easy visualization of all chips as a single mosaic in QGIS or similar tools.
:::

## KLT parameters leverage

### maxCorners & tile_size

In order to have a lower memory usage during KLT process, it is possible to define a tile size to process for KLT.

For example, a tile_size of 10000 for an image having a size of 20000 x 20000 pixels will result in 4 tiles to process.

In this context, the KLT process will look into each tile for `maxCorners`.

While an image of 20000 x 20000 pixels results in 4 equals tiles, an image of 20000 x 15000 pixels will also result in 4 tiles, but with different size, two of 10000 x 10000 pixels and two of 10000 x 5000 pixels.

The consequence is that the density for matching points will not be the same for each tile, the bigger tiles will have a lower matching point density than the smaller.

You should also consider that the image can contain empty parts where KLT will not find any matching point. So tiles having large empty parts will also result in a higher matching point density.

In order to avoid density differences in the final result, you can define a `tile_size` larger than the image with a high `maxCorners`, or a small `tile_size` and `maxCorners` in order to have tiles with almost the same size.

For example, for an image of 20000 x 15000 pixels, you should consider a `tile_size` of 20000 (1 tile), or 5000 (12 equal tiles).

## About shift by altitude plot

This output uses box plot to show statistics of KP on altitudes groups.

_The box extends from the first quartile (Q1) to the third quartile (Q3) of the data, with a line at the median. The whiskers extend from the box to the farthest data point lying within 1.5x the inter-quartile range (IQR) from the box. Flier points are those past the end of the whiskers. See [Box plot documentation [RD-12]](rd-12) for reference._

```
     Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
                  |-----:-----|
  o      |--------|     :     |--------|    o  o
                  |-----:-----|
flier             <----------->            fliers
                       IQR
```


## Input Data Relationships

Understanding how input files relate to each other:

### Image Compatibility

- **Monitored** and **Reference** images must be compatible:
  - Same footprint and resolution
  - Same coordinate system (EPSG)
  - Same pixel grid alignment

### Mask Usage

- **Mask** file must be compatible with the **monitored image**
- Used to exclude pixels from KLT feature detection
- Typical use cases: exclude water bodies, clouds, or invalid data areas
- Pixel value 0 = exclude from matching, non-zero = include

### DEM Usage

- **DEM** file must be compatible with the **reference image**
- Used to extract altitude values at key point locations
- Enables analysis of geometric errors vs terrain elevation
- Helps identify elevation-dependent systematic errors

## Troubleshooting

### Common Issues

#### Image Compatibility Errors

```
KariosException: Monitored image geo info not compatible with reference image
```

**Solution**: Ensure both images have:
- Same coordinate system (EPSG code)
- Same pixel resolution
- Same geographic extent
- Same grid alignment

#### Mask or DEM Compatibility Errors

```
KariosException: Mask geo info not compatible with monitored image
```

**Solution**: Ensure mask has the same geometry as monitored image.

### Performance Optimization

#### For Large Images

- Use smaller `tile_size` (e.g., 5000-10000 pixels)
- Reduce `maxCorners` per tile
- Enable `outliers_filtering: false` for faster processing

#### For High Accuracy

- Increase `maxCorners` for more key points
- Use smaller `matching_winsize` for precise matching
- Increase `qualityLevel` for better corner quality

## Advanced Features

### Large Shift Detection

When called with `--enable-large-shift-detection`, KARIOS can detect and compensate for large pixel offsets between images:

```bash
karios process monitored.tif reference.tif --enable-large-shift-detection
```

**Use cases**:

- Images with significant misalignment
- Coarse co-registration before fine matching
- Detection of systematic offsets

:::{important}
This is still an experimental feature that may use significant memory for large images.
:::

### Coarse-to-fine matching

With `--enable-coarse-to-fine`, KARIOS descends the image pyramid explicitly, seeding each level with a smoothed shift field fitted to the reliable points of the coarser one, instead of letting OpenCV recurse its own pyramid. It captures large displacements without the loss of key points along the data edges that a deep `maxLevel` causes.

```bash
karios process monitored.tif reference.tif --enable-coarse-to-fine
```

:::{important}
It cannot be combined with `--enable-large-shift-detection`, both remove a coarse displacement and would correct it twice. It does not support `laplacian_kernel_size: "auto"` either.
:::

### Laplacian tuning

The KLT tracker does not match the images themselves but their Laplacian, which keeps the edges and drops each sensor's gain and contrast. Three settings control it.

#### `--laplacian-power` (CLI)

Shape of the Laplacian rescale, from `0` to `1` (default `1`):

- `0`: the response is normalized and kept linear.
- `1`: it saturates to a near-binary edge / no-edge response. This is the most robust choice across sensors.
- In between: a signed power law interpolates, weak edges saturating sooner as the value grows.

```bash
karios process monitored.tif reference.tif --laplacian-power 0.5
```

#### `laplacian_kernel_size` (configuration file)

Aperture of the Laplacian filter. Larger kernels favor coarser structures and are less sensitive to noise. It takes one of:

- an integer, used for both images: `"laplacian_kernel_size": 7` (default);
- one size per image: `"laplacian_kernel_size": {"mon": 5, "ref": 9}`;
- `"auto"`: the **kernel sweep**.

With `"auto"`, each tile is matched with every pair of sizes among `3, 5, 7, 9, 11` for the monitored and the reference image, 25 combinations, and the pair with the highest inlier ratio is kept. It is useful when the two images have different resolutions or sharpness, at a higher processing time. The most frequent pair across tiles is reported in the logs and report.

#### `laplacian_invert_polarity` (configuration file)

Edges of images from different sensors can have opposite contrast, such as a dark road on a bright field in one image and the reverse in the other. This setting inverts the monitored pixels (`255 - pixel`) before the Laplacian:

- `false`: no inversion (default);
- `true`: always invert the monitored image;
- `"auto"`: **polarity sweep**. Each tile is matched twice, normal and inverted, and the run with the higher inlier ratio is kept. The processing time of the matching is doubled.

#### Reusing the auto results

When `laplacian_kernel_size` or `laplacian_invert_polarity` is `"auto"`, KARIOS writes into the output folder a copy of the configuration file where `"auto"` is replaced by the dominant choice found (for instance `{"mon": 7, "ref": 5}` and `false`). Use that copy with `--conf` on the next runs of the same image pair or sensor combination, to skip the sweep.

```json
"klt_matching": {
  "laplacian_kernel_size": "auto",
  "laplacian_invert_polarity": "auto"
}
```

### Global alignment (`karios align`)

`karios align` aligns a monitored image to a reference image before running `karios process`. It estimates a 2D homography (rotation, scale, translation and perspective) with SIFT feature matching and RANSAC, then refines it with ECC on Sobel gradient magnitudes.

```bash
karios align monitored.tif reference.tif --out ./aligned
```

It prints the estimated rotation, scale, translation, perspective and the inlier ratio, and writes to `--out`:

| File | Description |
|------|-------------|
| `<mon_stem>_global_aligned<ext>` | The monitored image aligned on the reference. It keeps its CRS and grid, over its corrected footprint. |
| `<mon_stem>_global_alignment.json` | The alignment estimated, to apply to other bands. |

Without georeferencing on both images, the result is rendered over its footprint in the reference, on a grid nested in the reference one and fine enough to keep the monitored resolution.

To compare the aligned image with the reference using `karios process`, both must be on the same grid. `karios align` prints the `gdalwarp` command that resamples the reference on the aligned image grid.

#### Aligning the other bands of a product

The bands of a product share its georeferencing error, but the alignment estimated on each one differs slightly, so bands aligned one by one no longer overlay. Estimate the alignment once and apply it to the other bands:

```bash
# estimate on B04 and warp B03 and B08 the same way
karios align B04.tif reference.tif --apply-to B03.tif --apply-to B08.tif --out ./aligned

# or later, from the saved alignment (no reference needed)
karios align B08.tif --load-transform ./aligned/B04_global_alignment.json --out ./aligned
```

A band on the same grid as the aligned one comes out on the same output grid, pixel for pixel. A band on another grid, such as a 20 m band of a 10 m product, is placed by its georeferencing, corrected the same way, and keeps its own grid and CRS.

#### Options

| Option | Type | Description |
|--------|------|------------|
| `--out` | PATH | Output directory [default: current directory] |
| `--apply-to` | FILE | Another band warped by the alignment estimated on the monitored image. Repeatable. |
| `--load-transform` | FILE | Apply a saved `<mon_stem>_global_alignment.json` instead of estimating one. `REFERENCE_IMAGE` is then left out. |
| `--sift-nfeatures` | INTEGER | Number of strongest SIFT keypoints kept per image. Lower it to reduce time and memory on large images. `0` is unlimited. |
| `--debug`, `-d`, `--no-log-file`, `--log-file-path` | | Logging options, as for `karios process`. |

Give either `REFERENCE_IMAGE` or `--load-transform`, not both.

### Resume Functionality

Skip KLT matching and reuse previous results:

```bash
karios process monitored.tif reference.tif --resume --out ./existing_results
```

**Requirements**:
- Previous KLT CSV results must exist in output directory
- Same images pair and configuration
