<!-- 
disable right sidebar 
it works in MD with myst_parser by enabling myst extension fieldlist
-->
:html_theme.sidebar_secondary.remove:

# Case studies

```{toctree}
:hidden:
:maxdepth: 2

prisma
sen2like
mss
```

This show how KARIOS is useful for our works.

<!-- SYNTAX 
https://sphinx-design.readthedocs.io/en/latest/grids.html#placing-a-card-in-a-grid
WARNING : We use card in grid-item instead of grid-item-card due to vertical spacing issue

For click
https://sphinx-design.readthedocs.io/en/latest/cards.html#clickable-cards

Grid works in MD thanks to myst extension colon_fence
-->

<!-- Grid start herer -->

:::::{grid} 2
<!-- item 1 -->
::::{grid-item}
:::{card} PRISMA
:link: prisma
:link-type: ref

```{todo}
Write short description
```
:::
::::
<!-- EO item 1 -->

<!-- item 2 -->
::::{grid-item}
:::{card} Sen2like
:link: sen2like
:link-type: ref

```{todo}
Write short description
```
:::
::::
<!-- EO item 2 -->

<!-- item 3 -->
::::{grid-item}
:::{card} MSS
:link: mss
:link-type: ref

```{todo}
Write short description
```
:::
::::
<!-- EO item 3 -->

:::::
<!-- EO Grid -->