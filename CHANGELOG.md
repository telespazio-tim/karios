# KARIOS CHANGELOG

## Unreleased

### New features

- **Coarse-to-fine matching** (`--enable-coarse-to-fine`, off by default): descends the image pyramid explicitly instead of letting OpenCV recurse its own.
  Each level runs without internal recursion and passes down a displacement field fitted to
  that level's reliable points, so key points near data edges inherit a usable starting guess
  rather than a broken one. On four scenes (SPOT5, Landsat-8/Sentinel-2, MSS terrain,
  Sentinel-2 10980²) it produced 20-100% more key points, far better coverage at data edges,
  and lower dispersion, for 1.4-1.6x the matching time and no extra memory. Not compatible
  with `laplacian_kernel_size: "auto"`, which is rejected when the matcher is built, nor with
  `--enable-large-shift-detection`: both remove a coarse displacement, so combining them would
  correct it twice.
- **Configurable pyramid depth** (`klt_matching.maxLevel`, default `3`): previously a source
  constant. In the default matching mode this is the parameter that most affects results, since
  the lost edge band scales as `(matching_winsize / 2) * 2**maxLevel`.
- **DEM elevation in the CSV output**: when a DEM is given, the `KLT_matcher_*.csv` file gets an
  `alt` column with the DEM elevation at each key point. Resuming (`--resume`) from a CSV written
  without a DEM adds the column.

- **Checkerboard mosaic** (`--mosaic-tile-size`, `0` by default, which disables it): mosaic
  at native resolution alternating tiles of the given size in pixel from the reference image, in
  blue, and the monitored image, in red, so misregistration shows as features broken at the tile
  edges. Each image is histogram equalized to 8 bit on its own, which gives both the same
  contrast whatever their sensor; zero fill and pixels hidden in the overview plot are black and
  left out of the histogram. Written as `05_mosaic.avif`, or lossless `.png` when Pillow lacks
  AVIF support, and displayed in a new *Mosaic* tab of the HTML report.
- **Color overlay** (`--generate-overlay`): the monitored image in the red channel and the
  reference in the green and blue ones, equalized like the mosaic, where aligned features are
  gray and shifted ones fringed in red and cyan. Written as `06_overlay.avif` and displayed in a
  new *Overlay* tab of the HTML report.

### Fix

- **`karios align` recovers large georeferencing errors between images of different
  resolutions**: SIFT ran on both whole images at native resolution, so a monitored image 7x
  finer than the reference had keypoints of details the reference cannot show, and its small
  footprint left most reference keypoints without a counterpart (166 of 5000 on a
  PhiSat/Sentinel-2 pair); ECC, which only corrects a few pixels, could not recover a
  georeferencing 5 km off either, and the highest ECC score then picked a -144° rotation from 5
  RANSAC inliers. With a geotransform prior, matching now runs at the coarser resolution on the
  reference cropped around the monitored footprint, a correlation search corrects the prior's
  translation as an extra ECC start, implausible estimates are rejected, and the others compare
  on common pixels. On that pair: 26/42 RANSAC inliers instead of 5/15, and `karios process` on
  the output measures 794 key points with a 0.6-0.7 px spread and a 0.1 px mean shift, against
  65 with 2.4-3.3 px for ECC from the georeferencing alone. The prior now also maps pixel
  centers, as OpenCV does, instead of the geotransform's pixel corners.
- **`karios align` keeps the monitored resolution**: the aligned monitored image was written on
  the reference's grid, so a 4 m PhiSat image aligned on a 30 m Sentinel-2 tile came out at
  30 m, downsampled 7x by bilinear interpolation. Both outputs now share a grid covering the
  monitored image's valid footprint, nested in the reference's grid at the reference pixel
  divided by the smallest integer keeping the monitored resolution (3.75 m for PhiSat). Unlike
  the reverted attempt, which covered the whole reference footprint (26267 px square for
  PhiSat), the grid spans the monitored data only (5904 x 6272 px). `karios align` now writes
  that image only: the reference and the alternative ECC candidates are no longer written, and
  the command prints the `gdalwarp` call that resamples the reference onto the output grid for
  `karios process`.
- **`karios align` handles 10 m references**: OpenCV's brute-force matcher refuses more than
  262143 descriptors, and a PhiSat scene on a 10 m Sentinel-2 crop has 484k; it would also have
  compared all 62k x 484k pairs, about 36 min per direction. Above 10^8 pairs matching now uses a
  FLANN KD-tree, approximate but absorbed by the Lowe ratio and cross-check filters.
- **`karios align` refines only the best starting point to convergence**: ECC ran its 200
  iterations from every starting point (RANSAC, prior, translation search), and the ones that
  lose ran them without converging, 51-69% of a 10 m PhiSat alignment. Every start now gets a
  25-iteration probe; the probes are compared by gradient correlation and only the winner is
  refined further.
- **`karios align` widens its search window when the georeferencing is further off**: matching
  only searched the reference around the monitored footprint, half its size wider on each side,
  so a larger georeferencing error gave a wrong result silently. A result with no plausible
  estimate, a gradient correlation under 0.2, or a shift beyond 75% of the margin now doubles the
  margin, up to the whole reference, keeping the best result of the windows tried; doubts left
  with the whole reference searched are logged. Only the window is read from the reference file.
- **`karios align` peak memory down from 7.2 to 2.1 GB** on a PhiSat scene and a 10 m
  Sentinel-2 tile, and 3 min 40 s instead of 6 min 20 s. OpenCV's SIFT doubles its input
  before building the pyramid, about 200 bytes per pixel: 6 GB for the 28 Mpx reference crop.
  Images wider than 2048 px are now detected by tiles read with a 128 px margin, keeping each
  tile's core keypoints; the thresholds being absolute, 99.98% of the keypoints lie within
  0.01 px of the whole image's. The reference is also stretched and equalized on its crop only,
  instead of the whole 120 Mpx tile.
- **ECC refinement in `karios align` corrected the wrong way**: ECC's warp maps the reference
  onto the pre-warped monitored image, and was composed without inverting it, so every
  refinement doubled its starting error instead of removing it (a start 2.5 px off ended 5 px
  off on the other side). Its mask also reached the edge of the data, where the gradients see
  the jump to the zero fill; eroding it raises the correlation of identical images from 0.77 to
  0.999. ECC now also works on the reference around the monitored footprint only, 8x faster on
  a footprint a quarter of the crop.
- **`karios align` uses the georeferencing whatever the CRS and grid orientation**: the prior was
  only built for two north-up images in the same CRS, so a PhiSat scene in WGS 84 on a rotated,
  mirrored grid ran blind on a UTM Sentinel-2 tile, matching the whole images for 6 RANSAC inliers
  of 841. The prior is now fitted on a grid of monitored pixels reprojected into the reference's
  pixels (within 0.9 px over that 22 km scene), and the monitored image is straightened by it
  into the reference's orientation before matching, as SIFT does not handle mirrored images.
- **Images in a geographic CRS or on a rotated grid no longer crash `karios process`**: the first
  geotransform term was taken as the pixel size in meters, so a PhiSat scene in WGS 84 on a
  rotated grid got -5.6e-05 "m", which inverted the circular error histogram range. A metric
  pixel size is now only read from a projected CRS on a north-up grid; other images are
  measured in pixels, unless `--input-pixel-size` is given. Their EPSG code is read too (it only
  was for projected CRS), so the key point GeoJSON is written, and the raster products copy the
  reference's full geotransform instead of dropping its rotation terms.
- **DEM plots no longer crash on matplotlib 3.8**: the shift-by-altitude plot passed `label` to
  `boxplot()`, which only accepts it from matplotlib 3.9, so every run with a DEM failed while
  generating reports.
- **`--no-value` now reaches the matching mask**: it previously only removed surviving key
  points and tinted the overview plot, so a product declaring no-data `0` while actually filled
  with another DN had features detected throughout the fill and reported an inflated valid-pixel
  count (99.93% where the true overlap was 66.63%). The values are now excluded from the KLT
  mask and from the valid-pixel statistic.
- **Key points are no longer lost at tile seams**: tiles did not overlap, so the matching window
  lacked support at every internal boundary. Tiles now read a margin of real neighbouring
  pixels, sized from the pyramid depth. Synthetic padding was measured and rejected: reflected
  or replicated content does not move consistently between the two images, so a window
  overlapping it scores worse than a truncated one.

- **`karios align` subcommand**: standalone command that warps the monitored image into the reference frame, keeping its resolution, and writes it georeferenced in the reference's CRS.
- **Configurable SIFT keypoint limit** (`karios align --sift-nfeatures`, `10000` by default, `0` = unlimited): keeps only the N strongest SIFT keypoints per image, to bound the brute-force matching time and memory on large images.

### Improvements

- **Global alignment algorithm overhaul**: replaced the original rotated-template sweep (±15° rotation, ±128 px translation search via `cv2.matchTemplate`) with a SIFT + homography RANSAC + ECC pipeline. The new algorithm:
  - estimates a full 3×3 homography (8 DOF — translation, rotation, scale, shear, perspective) instead of a 4-DOF rotation+translation+uniform-scale fit;
  - is far more robust on cross-sensor pairs thanks to CLAHE preprocessing, SIFT descriptors, Lowe ratio + mutual cross-check, and ECC refinement on Sobel gradient magnitudes (sensor-invariant);
  - drops the equal-size input constraint;
  - renders the warped monitored image onto the reference's pixel grid so both outputs share a geotransform.

## 2.1.1 [20260218]

### Fix

- **Correct axis label (TIGI-131)** - Fix axis labeling issue
- **Handle zero standard deviation in ZNCC computation** - Return NaN instead of raising exception when standard deviation is zero
- **Try to better handle some processing errors** - Improve error handling in processing pipeline
- **Update deps** - General dependency updates

## CI/CD
- **Add unit tests for KLT matcher, LargeOffsetMatcher, and ZNCC service** - Extensive unit test coverage for matcher components
- **Add e2e test**
- **Create GitHub Actions workflows for Ubuntu and Windows64** - Add CI support for both platforms
  - Test conda installation and run tests.

## 2.1.0 [20250812]

### New features

- Add generation of chip images of a selection of relevant KP using options `--generate-kp-chips`
- Add ZNCC score (`zncc_score`) for relevant KP in csv and JSON output.

### Fix

- Warning message during plot generation
- Missing conda env update instruction

## 2.0.0 [20240620]

### breaking changes

- DEM and mask files are now arguments, not options

### New features

- Installation: KARIOS can now be used outside of its directory by following installation procedure.
- Create API to use in another application
- Add shift by altitude groups plot
- Add KP geojson output
- Large shift detection (Experimental, know issue: use 11Go with 2 S2 at 10m resolution as reference and monitored image)

### Improvements

- Use [rich-click](https://ewels.github.io/rich-click/) in place of argparse
- Add processing config to output directory
- Add disclaimer in Geometric Error distribution figure about planimetric accuracy.
- Add input images geo information verification
- Refactor configuration by separating processing and plot configuration
- Attempt to better manage memory for large dataset

### Fix

- Fix northing reverse in statistics calculations and statistics usages in error distribution plot.
- Module name appears twice in log

### Documentation

- Add notice in Readme about CE90 accuracy.
- Update input images content recommendation

## 1.0.0 [20240119]

Initial version
