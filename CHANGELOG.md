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

### Fix

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

- **`karios align` subcommand**: standalone command that warps the monitored image onto the reference grid. Writes the primary aligned output plus one sibling per ECC-converged candidate for visual A/B comparison in QGIS.

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
