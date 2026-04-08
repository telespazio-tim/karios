(proba)=

# PROBA/CHRIS Processing

## Introduction
This case shows the PROBA/CHRIS geographic calibration. CHRIS images can be heavily distorted and are geographically inacurate.

Before running KARIOS, they need to be replaced roughly in the reference corresponding area, and also scaled/rotated in some cases.

The KARIOS results will then be used to perform a Thin Plate Spline (TPS) correction that will refine the pixel displacement.

The process is summarized in this chart : 

```{figure} chris.png
:name: chris
:width: 600px

CHRIS Process Summary
```

## 1. Coarse Registration​

A template matching algorithm can be used, followed by a re-alignment and scaling process.
With an ORB matching process, translation, scale and rotation values are estimated and applied to the CHRIS template. The result template will then we compared to a S2 reference with KARIOS.

```{figure} tm.png
:name: tm
:width: 600px

CHRIS Template Matching (left) and comparison with S2 (right)
```
```{figure} align.png
:name: align
:width: 600px

Alignment of the template over the reference
```
 
## 2. High Resolution KARIOS matching

Now that the image is aligned with the reference, KARIOS can be performed, initially at native resolution (18m) : 

```{figure} spain_18_before.png
:name: spain_18_before
:width: 600px

Geometric errors overview - CHRIS / S2 LiDAR before TPS correction (18m)
```

```{figure} spain_18_before_4.png
:name: spain_18_before_4
:width: 600px

Geometric errors distribution - CHRIS / S2 LiDAR before TPS correction (18m)
```

## 3. Low Resolution KARIOS matching

The results show that the matching has only been performed partially. The corners near the center are captured whereas the ones in the upper and lower part of the image are not detected as the difference is too high. Thus, the final result is not meaningful for the whole image, and the TPS correction should not applied as the process depends on the detected corners. A downsampling can be performed to allow better matching (bi-cubic). A resolution of 48m is a good compromise between matching and not lowering too much information.

```{figure} spain_48_before.png
:name: spain_48_before
:width: 600px

Geometric errors overview - CHRIS / S2 LiDAR before TPS correction (48m)
```

```{figure} spain_48_before_4.png
:name: spain_48_before_4
:width: 600px

Geometric errors distribution - CHRIS / S2 LiDAR before TPS correction (48m)
```

The corners now cover the entire image and show all the CHRIS deformations, though the point density is lower. The TPS correction can be applied by using a warping at full resolution.

## 3. Full Resolution​ Refinement

The TPS algorithm yields the following results :

```{figure} spain_18_after.png
:name: spain_18_after
:width: 600px

Geometric errors overview - CHRIS / S2 LiDAR after TPS correction (18m)
```

```{figure} spain_18_after_4.png
:name: spain_18_after_4
:width: 600px

Geometric errors distribution - CHRIS / S2 LiDAR after TPS correction (18m)
```

The RMSE has been reduced, but the final matching could be improved. A second TPS correction process can be applied to the result image : 

```{figure} spain_18_after_2.png
:name: spain_18_after_2
:width: 600px

Geometric errors overview - CHRIS / S2 LiDAR after additional TPS correction (18m)
```

```{figure} spain_18_after_4_2.png
:name: spain_18_after_4_2
:width: 600px

Geometric errors distribution - CHRIS / S2 LiDAR after additional TPS correction (18m)
```

The results now show a significant improvement in terms of metrics and the two image can now be compared with a full coverage at native resolution.