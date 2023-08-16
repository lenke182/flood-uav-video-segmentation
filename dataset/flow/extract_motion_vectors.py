# Extracts motion vectors from H.264 encoded videos and saves them

import numpy as np
import cv2
from mvextractor.videocap import VideoCap
import os
import sys
from flow.model import get_default_grid

height = 1072
width = 1920
block_size = 16

height_blocks = height // block_size
width_blocks = width // block_size

default_grid = get_default_grid()
        
# Transforms a list of block motion vectors for an image into a grid matrix
# and the inverse of that grid matrix
def transform_motion_vectors_to_grid_matrix(motion_vectors, H, W):
    grid = np.copy(default_grid)
    inv_grid = np.copy(default_grid)

    for m in motion_vectors:
        assert m[0] == -1

        size_x, size_y, src_x, src_y, dst_x, dst_y = m[1], m[2], m[3], m[4], m[5], m[6]

        assert (size_x == block_size and size_y == block_size)

        src_x_blocks, src_y_blocks = src_x // block_size, src_y // block_size
        dst_x_blocks, dst_y_blocks = dst_x // block_size, dst_y // block_size

        if 0 <= dst_x_blocks < width_blocks and 0 <= dst_y_blocks < height_blocks:
            grid[dst_y_blocks][dst_x_blocks][0] = (src_x_blocks * block_size+block_size//2)/W * 2 - 1
            grid[dst_y_blocks][dst_x_blocks][1] = (src_y_blocks * block_size+block_size//2)/H * 2 - 1

        if 0 <= src_x_blocks < width_blocks and 0 <= src_y_blocks < height_blocks:
            inv_grid[src_y_blocks][src_x_blocks][0] = (dst_x_blocks * block_size+block_size//2)/W * 2 - 1
            inv_grid[src_y_blocks][src_x_blocks][1] = (dst_y_blocks * block_size+block_size//2)/H * 2 - 1

    return grid, inv_grid



cap = VideoCap()

if len(sys.argv) == 2:
    videos = [
        sys.argv[1]
    ]
else:
    videos = [
        "florida-01.mp4",
        "florida-02.mp4",
        "florida-03.mp4",
        "florida-04.mp4",
        "florida-05.mp4",
        "florida-06.mp4",
        "florida-07.mp4",
        "florida-08.mp4",
        "florida-09.mp4",
        "texas-01.mp4",
        "florida-u.mp4"
    ]

# For each video
for v in videos:
    cap.open(v)

    v = v.split(".")[0]

    i = 0
    os.makedirs(os.path.join("frames", v, "images"), exist_ok=True)
    os.makedirs(os.path.join("frames", v, "grids"), exist_ok=True)
    os.makedirs(os.path.join("frames", v, "inv_grids"), exist_ok=True)

    while True:
        # For each frame
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()

        if not ret:
            break

        frame_filename = os.path.join("frames", v, "images"  , "{}.jpg".format(i))
        grid_filename = os.path.join("frames", v, "grids", "{}.npy".format(i))
        inv_grid_filename = os.path.join("frames", v, "inv_grids", "{}.npy".format(i))

        # If the frame wasn't already precomputed and extracted
        if not (os.path.exists(grid_filename) and os.path.exists(inv_grid_filename) and os.path.exists(frame_filename)):
            print("Extracting:", i, frame.shape, motion_vectors.shape)

            height = frame.shape[0]
            width = frame.shape[1]
            
            # Transform the motion vectors to grid matrices
            grid, inv_grid = transform_motion_vectors_to_grid_matrix(motion_vectors, height, width)

            # Saves the grid matrices and frame to disk
            if not os.path.exists(grid_filename):  
                np.save(grid_filename, grid)
            if not os.path.exists(inv_grid_filename):
                np.save(inv_grid_filename, inv_grid)
            if not os.path.exists(frame_filename):
                cv2.imwrite(frame_filename, frame)

        i += 1

    cap.release()
