# KARIOS CHANGELOG

## 2.1.0 [TODO]

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
