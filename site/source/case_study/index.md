<!-- 
disable right sidebar 
it works in MD with myst_parser by enabling myst extension fieldlist
-->
:html_theme.sidebar_secondary.remove:

# Case studies

```{toctree}
:hidden:
:maxdepth: 2

skysat
proba
mss
ccm
edap

```


:::::{grid} 2
::::{grid-item}
:::{card} SKYSAT / LiDAR Processing
:link: skysat
:link-type: ref

KARIOS matching process between Skysat and LiDAR high resolution images using downsampling and Laplacian optimization

:::
::::

::::{grid-item}
:::{card} PROBA/CHRIS Processing
:link: proba
:link-type: ref

CHRIS images registration using template matching and TPS low/native resolution refinement

:::
::::

::::{grid-item}
:::{card} MSS Processing
:link: mss_cs
:link-type: ref

Landsat MSS geometric correction using TPS

:::
::::

::::{grid-item}
:::{card} CCM Copernicus
:link: ccm
:link-type: ref

```{note}
Coming soon
```
:::
::::


::::{grid-item}
:::{card} EDAP Processing
:link: edap
:link-type: ref

```{note}
Coming soon
```
:::
::::

:::::
