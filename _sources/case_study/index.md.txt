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

<!-- Grid start herer -->

:::::{grid} 2
<!-- item 1 -->
::::{grid-item}
:::{card} SKYSAT / LiDAR Processing
:link: skysat
:link-type: ref

KARIOS matching process between Skysat and LiDAR high resolution images using downsampling and Laplacian optimization

:::
::::
<!-- EO item 1 -->

<!-- item 2 -->
::::{grid-item}
:::{card} PROBA/CHRIS Processing
:link: proba
:link-type: ref

CHRIS images registration using template matching and TPS low/native resolution refinement

:::
::::
<!-- EO item 2 -->

<!-- item 3 -->
::::{grid-item}
:::{card} MSS Processing
:link: mss_cs
:link-type: ref

Landsat MSS geometric correction using TPS

:::
::::
<!-- EO item 3 -->

<!-- item 4 -->
::::{grid-item}
:::{card} CCM Copernicus
:link: ccm
:link-type: ref

```{note}
Coming soon
```
:::
::::

<!-- EO item 4 -->

<!-- item 5 -->
::::{grid-item}
:::{card} EDAP Processing
:link: edap
:link-type: ref

```{note}
Coming soon
```
:::
::::
<!-- EO item 5 -->

::::
<!-- EO Grid -->