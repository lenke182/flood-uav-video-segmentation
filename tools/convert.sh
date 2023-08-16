#!/bin/sh

# Combines individual segmentation masks to a video file

module load ffmpeg
ffmpeg -framerate 25 -i "logs/$1/version_$2/frames/0/%d.png" -vcodec mpeg4 -b:v 2M "logs/$1/version_$2/frames/0.mp4"