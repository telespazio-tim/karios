# KARIOS CHANGELOG

## Unreleased

### New features

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
