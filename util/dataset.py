# Sources:
# - https://github.com/hszhao/semseg/blob/master/util/dataset.py

# Implements a Dataset instance for the segmentation task without frame interpolation

import os
import os.path
import numpy as np
from skimage import io

from torch.utils.data import Dataset
from flow.dataset import make_dataset

class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, predict_v_id="florida-01"):
        self.split = split

        # Dataset during training, validation and testing
        if self.split != "predict":
            self.data_list = make_dataset(split, data_root, data_list, frame_delta=0)
            self.length = len(self.data_list)
        else:
            # Dataset during inference
            self.v_id = predict_v_id
            self.length = len(os.listdir(os.path.join(self.data_root, "frames", self.v_id, "images")))

        print("Dataset Length: {}".format(self.length))
        self.transform = transform
        self.split = split
        self.data_root = data_root

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_path, v_id, f_index = self.data_list[index]

        # Load image
        image = self.get_frame(v_id, f_index)

        # Load label
        if self.split not in ['test', 'predict']:    
            label = io.imread(label_path)
        else:
            label = np.zeros((image.shape[0], image.shape[1]))

        # Apply data augmentation to data
        if self.transform is not None:
            image, label = self.transform(image, label)
            
        return image, label
    
    # Loads and returns a frame based on the video and frame id
    def get_frame(self, v_id, f_id):
        return io.imread(self.get_frame_path(v_id, f_id))

    # Returns the path for a frame
    def get_frame_path(self, v_id, f_id):
        return os.path.join(self.data_root, "frames", v_id, "images", str(f_id) + ".jpg") 

