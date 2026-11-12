(phisat_cs)=

# Phisat-2 Processing

## Introduction

In this study, KARIOS will be used to improve Phisat-2 geolocation error. The a priori geolocation accuracy, as estimated with product metadata is very low. A global shift of around 5km can be observed between the Phisat-2 product and the S2 reference (La Crau site). Furthermore, product shows strong geometric distortions (not due to terrain relief) that are mostly depending on band number. Delivered image geometric grid cannot be evaluated against a geometric reference by using the usual KARIOS process that is dedicated to analysis of sub pixel deformation (< 3 pixels), However, once bias is estimated, it is possible to preprocess data, and make grid comparison possible. This example will show how karios align command can help correcting this bias.

## 1. Absolute geolocation accuracy

One can note a strong variability of geolocation accuracy; the following salient points can be formulated:
* Geometric deformations are scene dependant. Error magnitude is varying from 2 km up to 4 km and the error type is mostly non-linear.
* Products can show very different geographic shift, even for products on the same area.
* Phisat-2 scene static bias against Sentinel-2 GRI has been estimated to be about 5300 m.

In the figure below, the Phisat-2 image (1) overlays an S2 image (2) and an important shift of about 5300 m is observed. It is worth noting that the shift in not constant inside the image.

```{figure} phisat_shift.png
:name: phisat_shift
:width: 600px

Strong geometric mis-registration between Phisat-2 band 3 (RED) / S2 band B04 images.
```

(1)	PHISAT-2_L1_000004343_20260109105212_20260109105216_3F61AA89 (band 3)
(2)	S2A_OPER_MSI_L1C_TL_MPC__20161018T120000_A000108_T31TFJ (band B04)

```{figure} phisat_distortion.png
:name: phisat_distortion
:width: 600px

Strong distortions between Phisat-2 – RGB (La Crau) 4.75m resolution rescaled at S2 full resolution (10m) and S2 – RGB (La Crau) 10m resolution rescaled at full resolution.


```
The objective is to build a generic model that will refine the input image grid with sophisticated warping functions.
First, a coarse co-registration model has been implemented to improve geolocation by estimating a homography between the Phisat-2 product and a Sentinel-2 reference. This process is performed by karios align command. Then, the result is used to perform a fine co-registration using TPS correction, also with the same Sentinel-2 reference.

```{figure} global_process.png
:name: global_process
:width: 300px

Global process.

```

## 2. SIFT Matching with karios align


A few experiments / testing confirmed that it is relevant to use the SIFT method for Phisat-2 image data. Note that the SIFT method estimates a homography transformation matrix, meaning that a geometric transformation expressed as a combination of translation, rotation and uniform scaling is expected to be appropriate.


### 2.1 Method

The objective of the SIFT algorithm is to automatically co-register the Phisat-2 image. This technique is based on matching the product with a S2 reference on the same area, so the resulting image can be usable by karios process command. karios process is initially not usable as the image is too shifted and distorted. The main idea is to perform a coarse co-registration to allow a relevant matching. 

The process is performed on the full Phisat-2 product on band 0 (PAN). Then, the resulting homography is applied to all bands. The reference S2 has been selected so the image extent is larger and contains the area of interest.

```{figure} sift_process.png
:name: sift_process
:width: 600px

Matching and Co-registration process.
```

### 2.2 Pre-processing

The Phisat-2 product is not pre-processed, the whole band content can used as input to the process, in its original projection.
Note that for showing the KARIOS results, the resulting product is reprojected to EPSG:32631 - WGS 84 / UTM zone 31N; 4.18m pixel spacing is reached by using an average convolution kernel.

The S2 product is a L1C S2 product.
The S2 product is cropped on a ROI which is much larger and that contains the Phisat-2 ROI. The extent should be a 10km extended area centred on the original Phisat-2 product. This estimation of 10km is subject to change, as some products can have a much larger shift.

### 2.3 Image Matching

The matching algorithm is based on OpenCV’s SIFT matcher. It uses the results of the detectAndCompute method (i.e. key points, descriptors) which is applied to the monitored product and the reference (RD-24). Each key-point is associated to a descriptor (vector of 128 values for SIFT) that encodes the local appearance in terms of gradients, intensity. Two similar descriptors should point at the same real-world point.

The model then estimates a homography representing the difference in scale, translation and rotation between the two images.

As an example, for band 1 the result homography matrix is:

```{figure} matrix.png
:name: matrix
:width: 600px
```

This matrix shows:
* A translation of 943 m for x and 887 m for y (1295 m globally)
* A scaling of det(H) = 0.94
* A rotation of arctan(-0.2219/0.9755) = -12.8°


### 2.4 Results

The co-registered image has been compared to the S2 reference with KARIOS.
A good matching is achieved, with an RMSE accuracy close to 7m on band 1. An internal distortion is still visible inside the image, which is non-systematic and can be attributed to terrain elevation (Phisat-2 image is not ortho-rectified).

```{figure} sift_res1.png
:name: sift_res1
:width: 600px

KARIOS error overview between Phisat-2 band 3 (RED) co-registered product and S2 reference (band B04).
```

```{figure} sift_res2.png
:name: sift_res2
:width: 600px

KARIOS error overview between Phisat-2 co-registered product and S2 reference.
```

## 3 Thin Plate Spline correction using KARIOS 

### 3.1 Method

After having roughly co-registered the Phisat-2 images, a finer co-registration can be achieved using KARIOS and the TPS algorithm.
TPS regression is a specific case of RBF regression (RD-19) where the RBF is the thin plate radial basis function for r²log®. A fine co-registration can be performed to reach an accuracy within 1 pixel.
The algorithm has been applied to the results of the manual georeferencing and the SIFT matching.


### 3.2 KARIOS outputs

The KARIOS TPS algorithm improves significantly image precision; when applied RMSE accuracy reaches 5-7m for manual and 2m for SIFT.

```{figure} tps_res1.png
:name: tps_res1
:width: 600px

Error overview after SIFT georeferencing without (left images) and with (right images) KARIOS TPS correction.
```

```{figure} tps_res2.png
:name: tps_res2
:width: 600px

Accuracy statistics after SIFT georeferencing without (left) and with (right) KARIOS TPS correction.
```

### 3.3 KARIOS performance results

The SIFT and manual georeferencing both allow to have an image which is now comparable with the reference with KARIOS. After using the KARIOS detected GCPs with a TPS algorithm, the accuracy in now close to the Phisat-2 pixel resolution.
The SIFT georeferencing with TPS correction show the best results, with an excellent matching between the corrected Phisat-2 product and the S2 reference and a RMSE of 2m.


```{figure} perfo.png
:name: perfo
:width: 600px

Geolocation accuracy before and after processing (RED band).
```

## Conclusion 

The results show a significant improvement in the global RMSE, in particular in the south of France and Poland dataset. For Greenland, the improvement is more limited as the mountainous terrain and the high ice coverage hinders the possibility to have a good KARIOS matching. Many parameters have been studied (number of cells, etc) with most of them being KARIOS parameters, which plays a key role in the geometric process.

```{include} ../_includes/endnote.md