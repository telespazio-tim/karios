(mss)=

# MSS

## Introduction
This case shows the Landsat MSS geometric processing and KARIOS results.

```{figure} mss.png
:name: mss
:width: 600px

MSS Processing overview
```

## 1. MSS Geometric Correction

The objective is to apply a poly-harmonic splines geometric transformation model to MSS data to account for local deformation.
The poly-harmonic splines are a linear combination of Radial Basis Functions (RBFs) plus a 2nd degree polynomial term :

```{figure} formule1.png
:name: formule1
:width: 300px

```
The model is applied for co registration of MSS Data to a common reference map, and is calibrated by using reference GCP set defined for every cell.

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