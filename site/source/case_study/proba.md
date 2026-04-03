(proba)=

# PROBA Processing

## Introduction
This case shows the PROBA/CHRIS geographic calibration. CHRIS images can be heavily distorted and are geographically inacurate.

Before running KARIOS, they need to be replaced roughly in the reference corresponding area, and also scaled/rotated in some cases.

## Pre-processing

A template matching algorithm can be used, followed by a re-alignment and scaling process.
The result template will then we compared to a S2 reference with KARIOS.

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