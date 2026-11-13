#!/bin/bash
# usage: append_end.sh in.mp4 out.mp4 -- fade into the end screen (0.8 s crossfade)
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$1")
OFF=$(python3 -c "print(round($D-0.8,2))")
ffmpeg -loglevel error -y -i "$1" -i "$(dirname "$0")/logo/end2.mp4" \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.8:offset=$OFF,format=yuv420p[v]" \
  -map "[v]" -c:v libx264 -crf 18 -r 15 -movflags +faststart "$2"
