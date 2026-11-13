# KARIOS demo videos

Generators for the videos shown on the website (`site/source/cookbook/videos.md`).
All videos are 1920x1080, 15 fps, H.264, and end with the Telespazio end screen.

Requirements: Python 3 + Pillow, ffmpeg, DejaVu fonts, a KARIOS 3.0 install (`karios` on the path),
GDAL (`gdalwarp`), headless Chrome (report screenshots), QGIS + `xvfb-run` (scenario 2 only).
Set `VIDEO_WORK=<dir>` to the scratch directory (default `work/`).

| Script | Output |
|---|---|
| `mkintro.py out.mp4` | KARIOS 3.0 introduction (slides; run `prep_assets.py <issues>/images` first) |
| `mkdemo.py out.mp4` | quickstart (terminal + result plots) |
| `mkreport.py out.mp4 $VIDEO_WORK` | HTML report tour, zooms on mosaic/report/chips |
| `mkscen.py <1-6> outdir` | scenario videos |
| `runall.sh`, `run2.sh` | run `karios process` for the scenarios into `$VIDEO_WORK/demo/<n>` |
| `qgis/shot.py` | QGIS screenshot of the key points (scenario 2) |
| `mkend.sh`, `append_end.sh in out` | build the end screen / append it with a crossfade |

Inputs are local datasets whose paths are hard-coded in the scripts (`2025-KARIOS_DEMO`, `lacraux`,
`images/` of the 3.0.0 issue folder): adapt them to your layout. Screenshots of the HTML report
(`chips6/`, `fr/`, `qgis/` in the work dir) were made with headless Chrome from the report folders.

Logo: `logo/wiki.svg` is the unmodified Telespazio logo (Wikimedia Commons), centred on white with
clear space; no recolouring or distortion. Check against the Telespazio brand rules before release.
