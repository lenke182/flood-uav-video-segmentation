#!/bin/bash

# Generates the dataset based on two YouTube videos

set -e

yt-dlp -o florida https://www.youtube.com/watch?v=VF1CMbPlmPo
yt-dlp -o texas https://www.youtube.com/watch?v=SybD-lXqYR8
./ingress_new_video.sh florida
./ingress_new_video.sh texas
