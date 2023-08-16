#!/bin/sh

# Encodes a video file with the correct H.264 encoding

#module load ffmpeg
ffmpeg -i $1.webm -c:v libx264 -x264opts bframes=0 -partitions none -filter:v fps=25,scale=1920x1072 $1.mp4
export PYTHONPATH="../.."
python3 extract_motion_vectors.py $1.mp4
