# Sources:
# - https://github.com/hszhao/semseg/blob/master/util/util.py

import torch
import numpy as np
from PIL import Image

# Saves the mIoU metric in a class to average the results
# Source: https://github.com/hszhao/semseg/blob/master/util/util.py
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Poly learning rate scheduler
# Source: https://github.com/hszhao/semseg/blob/master/util/util.py
def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


# Computes the intersection and union for the mIoU metric
# Source: https://github.com/hszhao/semseg/blob/master/util/util.py
def intersectionAndUnion(output, target, K, ignore_index=255):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


# Computes the intersection and union for the mIoU metric on the GPU
# Source: https://github.com/hszhao/semseg/blob/master/util/util.py
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

# Returns a segmentation mask with color
# Source: https://github.com/hszhao/semseg/blob/master/util/util.py
def colorize(gray, palette):
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color
