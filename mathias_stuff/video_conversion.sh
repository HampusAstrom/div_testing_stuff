#!/bin/bash
# call this script with 'bash -i scripts/...'

source ~/.bashrc
# Delete since we can't do ros stuff otherwise
rm ./deps/limbo/exp/blackdrops

mon launch blackdrops video_export.launch

cd ./video
ffmpeg -framerate 15 -i frame%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
rm -r frame*.jpg
cd ..

