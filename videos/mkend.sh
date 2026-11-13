#!/bin/bash
# 4 s Telespazio end screen (logo/end2.png = logo/wiki.svg, unmodified, centred on white) -> logo/end2.mp4
ffmpeg -y -loop 1 -framerate 15 -i logo/end2.png -t 4 -c:v libx264 -pix_fmt yuv420p -crf 18 logo/end2.mp4
