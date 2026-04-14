(mss_cs)=

# MSS Processing

## Introduction

The KARIOS tool can be easily integrated into a more complex geometric calibration processing in charge of Landsat MSS L1C product refinement [RD-11](rd-11).
This case shows how KARIOS results are used to estimate the deformation model, subsequently used for warping to Sentinel 2 reference image grid. Furthermore, KARIOS is used to check the quality of resulting L1C refined image. An in depth analysis, based on large dataset confirms that the proposed approach is relevant.

The land monitoring community expects consistent and harmonised long-term datasets, in order to derive Essential Climate Variables (ECV).
Within this context, in complement to Thematic Mapper (TM) and Enhanced Thematic Mapper (ETM) data, the ESA archive includes also Landsat Multi
Spectral Scanner (MSS) data, which represent an outstanding source of historical data [RD-1](rd-01).
In the last decades, many efforts spent in the consolidation of this ESA archive, focusing on raw data repatriation, definition of new product type and bulk
processing of the full archive (ESA SLAP [RD-2](rd-02), [RD-3](rd-03), [RD-4](rd-04)).
Nonetheless, in the era of data cube, the MSS data is now requiring challenging algorithm developments to ensure that all threshold requirements, as defined
by the CEOS Analysis Ready Data For Land (CARD4L) Surface Reflectance (SR) Product Family Specifications (PFS), ([RD-5](rd-05)), are met.

```{figure} mss.png
:name: mss
:width: 600px

MSS Processing overview
```

## 1. MSS Geometric Correction

The status is that the geometric accuracy of delivered Landsat
MSS ESA/SLAP products is not sufficient to reach CEOS ARD
compliancy at threshold level, because their multi temporal
registration accuracy is not sub-pixel. The bad precision of the
geolocation is a major contributor to uncertainty loss, It
prevents to reach 0.5 pixel RMSE multi temporal accuracy.

The current ESA-MSS geo-processing is correct and include
state of art geometric calibration algorithm (bundle block
adjustment).

The objective is to apply a poly-harmonic splines geometric transformation model to MSS data to account for local deformation. Also, a way forward might be to correct for local geometric distortions by using Radial Basis Functions (RBF) as proposed
in ([RD-6](rd-06), [RD-7](rd-07)).
The poly-harmonic splines are a linear combination of RBFs plus a 2nd degree polynomial term :

```{figure} formule1.png
:name: formule1
:width: 300px

```
Where :

* N represents the total number of cells
* Ci the cell center coordinate
* Wi the weighting factor to be estimated
* The model is applied for co-registration of MSS L1C data to a common reference map.
* The model is calibrated by using reference GCP set (control point) defined for every cell (ci)
* Cells are selected into input images, the number / dimension of cells play an important role in the final result.

The following chart summarizes the RBF chain : 

```{figure} rbf.png
:name: rbf
:width: 300px

```

* Data Preparation: clipping over the same geo extent
* Matching: Collect Dense GCPs by using KARIOS applied on image twin (MSS Image, S2 image)
* Poly Harmonic Model Calibration: process GCPs grid, select GCPs relevant for calibration, and apply least square.
* Warping: transform input MSS image with polyharmonic model and generate output MSS Geo re-calibrated product
* Validation: use KARIOS, to check co-registration between S2 reference and output image.

## 2. Example results

These plots show the difference between the KARIOS results before and after the correction, against a S2 reference image, for products located in South of France, Greenland and Poland :

### 2.1 South of France

```{figure} toulousemss.png
:name: toulousemss
:width: 600px

Geometric errors overview - Landsat MSS / S2 (South of France)
```
```{figure} toulousemss2.png
:name: toulousemss2
:width: 600px

DY pixel shift (mean/STD) - Landsat MSS / S2 (South of France)
```
```{figure} toulousemss3.png
:name: toulousemss3
:width: 600px

Geometric errors distribution - Landsat MSS / S2 (South of France)
```
```{figure} toulousemss4.png
:name: toulousemss4
:width: 600px

Radial error shift by altitude distribution - Landsat MSS / S2 (South of France)
```

### 2.2 Greenland

```{figure} greenland1.png
:name: greenland1
:width: 600px

Geometric errors overview - Landsat MSS / S2 (Greenland)
```
```{figure} greenland2.png
:name: greenland2
:width: 600px

DY pixel shift (mean/STD) - Landsat MSS / S2 (Greenland)
```
```{figure} greenland3.png
:name: greenland3
:width: 600px

Geometric errors distribution - Landsat MSS / S2 (Greenland)
```
```{figure} greenland4.png
:name: greenland4
:width: 600px

Radial error shift by altitude distribution - Landsat MSS / S2 (Greenland)
```

### 2.3 Poland

```{figure} poland1.png
:name: poland1
:width: 600px

Geometric errors overview - Landsat MSS / S2 (Poland)
```
```{figure} poland2.png
:name: poland2
:width: 600px

DY pixel shift (mean/STD) - Landsat MSS / S2 (Poland)
```
```{figure} poland3.png
:name: poland3
:width: 600px

Geometric errors distribution - Landsat MSS / S2 (Poland)
```
```{figure} poland4.png
:name: poland4
:width: 600px

Radial error shift by altitude distribution - Landsat MSS / S2 (Poland)
```

## 3. Global results

These plots show the circular error results for all tested products (about 100 products per site) before and after correction :

### 3.1 South of France

```{figure} global_toulouse.png
:name: global_toulouse
:width: 600px

Circular error plot - All products - Landsat MSS / S2 (South of France)
```

### 3.2 Greenland

```{figure} global_greenland.png
:name: global_greenland
:width: 600px

Circular error plot - All products - Landsat MSS / S2 (Greenland)
```

### 3.3 Poland

```{figure} global_poland.png
:name: global_poland
:width: 600px

Circular error plot - All products - Landsat MSS / S2 (Poland)
```

```{warning}

The scale is different in the before/after plots.
```

## Conclusion 

The results show a significant improvement in the global RMSE, in particular in the south of France and Poland dataset. For Greenland, the improvement is more limited as the mountainous terrain and the high ice coverage hinders the possibility to have a good KARIOS matching. Many parameters have been studied (number of cells, etc) with most of them being KARIOS parameters, which plays a key role in the geometric process.

```{include} ../_includes/endnote.md