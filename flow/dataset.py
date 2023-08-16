# Implements the PyTorch Dataset for a frame interpolation model
# Implementation is based on Jain and Gonzalez (https://arxiv.org/pdf/1803.07742.pdf)
# Sources:
# - https://github.com/hszhao/semseg/blob/master/util/dataset.py

import torch
import os
from skimage import io
import numpy as np
import random
from torch.utils.data import Dataset
from flow.model import get_default_grid

# Returns a list of dataset items from a txt file
# Source: https://github.com/hszhao/semseg/blob/master/util/dataset.py
def make_dataset(split='train', data_root=None, data_list=None, frame_delta=25):
    image_label_list = []
    
    list_read = open(data_list).readlines()
    
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')

        if len(line_split) != 4:
            raise RuntimeError("Image list file read line error : " + line + "\n")
        
        label_name = os.path.join(data_root, line_split[0])
        video_id = line_split[1]
        frame_id = int(line_split[2])
        
        if frame_id < frame_delta//2:
            print("Skipping file {} with frame_id {} for video {}".format(label_name, frame_id, video_id))
            continue
        
        image_label_list.append((label_name, video_id, frame_id))
    
    print("Checking image&label pair {} list done!".format(split))
    
    return image_label_list

# Dataset class for a frame interpolation model
class FlowData(Dataset):
    def __init__(self, split='train', type="l", data_root=None, data_list=None, transform=None, frame_delta=8, no_warp=False, predict_v_id="florida-01", no_random_frame_delta=False):
        self.split = split
        self.transform = transform
        self.split = split
        self.data_root = data_root
        self.type = type
        self.frame_delta = frame_delta
        self.no_warp = no_warp
        self.no_random_frame_delta = no_random_frame_delta

        # Dataset during training, validation and testing
        if self.split != "predict":
            self.label_list = make_dataset(split, data_root, data_list, frame_delta=frame_delta)
            self.length = len(self.label_list)
        else:
            # Dataset during inference
            self.video_id = predict_v_id
            self.length = len(os.listdir(os.path.join(self.data_root, "frames", self.video_id, "images")))//self.frame_delta
        
        print("Dataset Length: {}".format(self.length))

        self.default_grid = get_default_grid()

    def __len__(self):
        return self.length

    # Returns a single dataset item that can contain the following items:
    # - preceding frame
    # - succeeding frame
    # - current frame (image of the labelled frame)
    # - grid matrices (forward & backwards)
    # - label
    # - metadata like frame id, amount of forward/backwards warping
    def __getitem__(self, index):
        if self.split != "predict":
            label_path, v_id, f_index = self.label_list[index]
        else:
            label_path = None
            v_id = self.video_id
            f_index = index * self.frame_delta


        # Select a preceding and succeeding frame
        if self.split in ["test", "val"]:
            frame_delta_l = random.Random(index).randrange(1,self.frame_delta)
            frame_delta_r = self.frame_delta - frame_delta_l
        else:
            # Train
            if self.no_random_frame_delta:
                frame_delta_l = self.frame_delta//2
                frame_delta_r = self.frame_delta//2
            else:
                frame_delta_l = random.randrange(1,self.frame_delta)
                frame_delta_r = self.frame_delta - frame_delta_l   

        assert frame_delta_l > 0 or frame_delta_r > 0

        # Load current frame
        if self.split == "train":
            frame_current = self.get_frame(v_id, f_index)
        else:
            frame_current = None
        

        if self.type != "gt":
            if self.split == "predict":
                f_prev_index = f_index
                f_next_index = f_index + self.frame_delta
            else:
                f_prev_index = f_index - frame_delta_l    
                f_next_index = f_index + frame_delta_r
            
            # Ensure that all previous and succeeding frames exist
            f_prev_index_real = f_prev_index
            while not os.path.exists(self.get_frame_path(v_id, f_prev_index_real)) or \
                not os.path.exists(self.get_grid_path(v_id, f_prev_index_real, "grids")) or \
                not os.path.exists(self.get_grid_path(v_id, f_prev_index_real, "inv_grids")):
                f_prev_index_real += 1
            frame_prev = self.get_frame(v_id, f_prev_index_real)
            f_next_index_real = f_next_index
            while not os.path.exists(self.get_frame_path(v_id, f_next_index_real)) or \
                not os.path.exists(self.get_grid_path(v_id, f_next_index_real, "grids")) or \
                not os.path.exists(self.get_grid_path(v_id, f_next_index_real, "inv_grids")):
                f_next_index_real -= 1
            frame_next = self.get_frame(v_id, f_next_index_real)


            # Load grid matrices if frame interpolation with block motion vectors is used
            if not self.no_warp:
                mvs_left = []
                mvs_right = []
                if self.split == "predict":
                    # During inference
                    for i in range(self.frame_delta-1):
                        mvs_left.append(self.load_grid(v_id, f_index+i+1))

                    for i in range(self.frame_delta-1):
                        mvs_right.append(self.load_inv_grid(v_id, f_index+i+1))

                    mvs_right.reverse()
                else:
                    # During training, validation and testing
                    for i in range(frame_delta_l):
                        grid_index = f_index-frame_delta_l+i+1
                        if grid_index > f_prev_index_real:
                            mvs_left.append(self.load_grid(v_id, grid_index))
                        else:
                            mvs_left.append(self.default_grid)
                    
                    while len(mvs_left) < (self.frame_delta-1):
                        mvs_left.append(self.default_grid)

                    for i in range(frame_delta_r):
                        grid_index = f_index+i+1
                        if grid_index <= f_next_index_real:
                            mvs_right.append(self.load_inv_grid(v_id, grid_index))
                        else:
                            mvs_right.append(self.default_grid)
            
                    mvs_right.reverse()

                    while len(mvs_right) < (self.frame_delta-1):
                        mvs_right.append(self.default_grid)
            else:
                mvs_left = None
                mvs_right = None
        else:
            frame_prev = None
            frame_next = None
            mvs_left = None
            mvs_right = None

        # Load label
        if self.type != 'u' and self.split != "predict":    
            label = io.imread(label_path)
        else:
            label = None

        # Data augmentation
        if self.transform is not None:
            frame_current, frame_prev, frame_next, mvs_left, mvs_right, label = self.transform(frame_current, frame_prev, frame_next, mvs_left, mvs_right, label)

        # Save all data into dictionary
        result = {}

        if frame_current is not None:
            result["frame_current"] = frame_current
        if frame_prev is not None:
            result["frame_prev"] = frame_prev
        if frame_next is not None:
            result["frame_next"] = frame_next
        if mvs_left is not None:
            result["mvs_left"] = mvs_left
        else:
            result["mvs_left"] = [torch.zeros(1)]*(self.frame_delta-1)
        if mvs_right is not None:
            result["mvs_right"] = mvs_right
        else:
            result["mvs_right"] = [torch.zeros(1)]*(self.frame_delta-1)
        if label is not None:
            result["label"] = label

        if self.split == "predict":
            result["frame_id"] = f_index
        

        result["left_index"] = frame_delta_l
        result["right_index"] = frame_delta_r

        return result

    # Loads and returns a frame based on the video and frame id
    def get_frame(self, v_id, f_id):
        return io.imread(self.get_frame_path(v_id, f_id))

    # Returns the path for a frame
    def get_frame_path(self, v_id, f_id):
        return os.path.join(self.data_root, "frames", v_id, "images", str(f_id) + ".jpg") 

    # Loads and returns a grid matrix (forward)
    def load_grid(self, v_id, i):
        return self._load_grid(v_id, i, "grids")

    # Loads and returns a grid matrix (backwards)
    def load_inv_grid(self, v_id, i):
        return self._load_grid(v_id, i, "inv_grids")

    # Returns the path for a grid matrix
    def get_grid_path(self, v_id, i, name):
        return os.path.join(self.data_root, "frames", v_id, name, str(i) + ".npy")

    # Loads a grid matrix from disk
    def _load_grid(self, v_id, i, name):
        return np.load(self.get_grid_path(v_id, i, name)).astype('float32')
