(config)=

# Influence of settings

This section presents the parameters that can be found and modified on the processing configuration_file.yaml file that can be added in the command line input (--conf). Otherwise, a default configuration file will be applied. These parameters are critical for the KLT matching and can greatly change the outputs.

## 1. KLT parameters

### 1.1 minDistance

Minimum Euclidean distance (in pixels) between detected feature points (corners).

With small values, features can be very close to each other which implies more corners, especially in textured areas. There could be a risk of redundancy and more similar points (can be good for dense matching, but heavier to track).

For larger values, features are more spread out. There will be fewer points, more spatially uniform coverage, which is good enough for global alignment and computational time, but less interesting for very fine local deformation.

### 1.2 blocksize

Size of the neighborhood (in pixels) used for computing the covariance matrix for corner detection.

With small values, the model will be sensitive to very fine details and noise which is good for high‑resolution, sharp imagery.

For larger values, corners are more stable and less noisy, but it might miss very small features.

### 1.3 maxCorners

Maximum number of corners (features) to detect.

Can be used to reduce the number of detected points for dense areas when computational power / processing time is an issue.

### 1.4 matching_winsize

Size of the search window (in pixels) used when matching/tracking features between images. 

With small windows, the process will be faster and precise when images are already quite aligned. But it might fail if displacement is too large.

For larger windows, the model will be able to handle large displacements between images, making it more robust when initial alignment is rough. But it gets more computationally expensive and can increase risk of mismatches if texture is repetitive.

### 1.5 qualityLevel

Threshold for selecting strong corners, expressed as a fraction of the best corner response (between 0 and 1). 

With high values, only very strong corners are kept which provides fewer but more reliable features.

With low values, there will be many more corners, including weak ones which could be provide noisy but denser outputs.

### 1.6 xStart

Horizontal offset (in pixels) from which to start processing in the image.

### 1.7 tile_size

Size (in pixels) of the processing tile, usually in width/height. 

Large images can be too big to process fully, so this allows to process the image in (tile_size) x (tile_size pixel) chunks. Feature detection and matching are done per tile, which reduces memory usage and can parallelize processing.

Smaller tiles means less memory, but more edge effects and overhead.
Larger tiles means better global context, but heavier on RAM.

### 1.8 laplacian_kernel_size

Kernel size for the Laplacian filter. 

Used for edge enhancement, focus/texture measure and pre‑filtering before feature detection.

Smaller kernel will make the model more sensitive to fine edges and noise, whereas large kernels will give 
a smoother response and emphasize broader structures, better for high resolution imagery.

### 1.9 outliers_filtering

Kernel size for the Laplacian filter. 

Whether to filter out mismatched or inconsistent feature matches after KLT tracking.

When true, bad matches (e.g. due to occlusions, parallax, moving objects, repetitive patterns) are removed.
When false, all matches are kept, including bad matches, which can be risky for the model.

## Accuracy analysis

confidence_threshold : parameter that controls the points which are used for the circular error plot and stats.

A higher value will provide more relevant points with a high confidence value, which can lead to more precise statistics but visually less smooth with fewer points.

A lower value will take a lot of points in consideration which can include less relevant results but with nicer display.

This value does not impact the tracking.

```{include} ../_includes/endnote.md