# Implements data augmentation for the segmentation task with frame interpolation
# Sources:
# - https://github.com/hszhao/semseg/blob/master/util/transform.py

import random
import math
import numbers
import collections
import cv2
import torch


# Implements the chaining of data augmentation techniques in a list
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class Compose(object):
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        for t in self.segtransform:
            frame_current, frame_prev, frame_next, mvs_left, mvs_right, label = t(frame_current, frame_prev, frame_next, mvs_left, mvs_right, label)
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label

# Transpose the input data from numpy to a tensor
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class ToTensor(object):
    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        if frame_current is not None:
            frame_current = torch.from_numpy(frame_current.transpose((2, 0, 1)).copy())
            if not isinstance(frame_current, torch.FloatTensor):
                frame_current = frame_current.type(torch.float32)
        if frame_prev is not None:
            frame_prev = torch.from_numpy(frame_prev.transpose((2, 0, 1)).copy())
            if not isinstance(frame_prev, torch.FloatTensor):
                frame_prev = frame_prev.type(torch.float32)
        if frame_next is not None:
            frame_next = torch.from_numpy(frame_next.transpose((2, 0, 1)).copy())
            if not isinstance(frame_next, torch.FloatTensor):
                frame_next = frame_next.type(torch.float32)
        
        map_func = lambda m: torch.from_numpy(m).type(torch.float32)
        if mvs_left is not None:
            mvs_left = list(map(map_func, mvs_left))
        if mvs_right is not None:
            mvs_right = list(map(map_func, mvs_right))

        if label is not None:
            label = torch.from_numpy(label)
            if not isinstance(label, torch.LongTensor):
                label = label.long()
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label


# Normalize the input data with mean and standard deviation
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class Normalize(object):
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        if self.std is None:
            if frame_current is not None:
                for t, m in zip(frame_current, self.mean):
                    t.sub_(m)
            if frame_prev is not None:
                for t, m in zip(frame_prev, self.mean):
                    t.sub_(m)
            if frame_next is not None:
                for t, m in zip(frame_next, self.mean):
                    t.sub_(m)
        else:
            if frame_current is not None:
                for t, m, s in zip(frame_current, self.mean, self.std):
                    t.sub_(m).div_(s)
            if frame_prev is not None:
                for t, m, s in zip(frame_prev, self.mean, self.std):
                    t.sub_(m).div_(s)
            if frame_next is not None:
                for t, m, s in zip(frame_next, self.mean, self.std):
                    t.sub_(m).div_(s)
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label


# Resize the input to a specific size
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class Resize(object):
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        if frame_current is not None:
            frame_current = cv2.resize(frame_current, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        if frame_prev is not None:
            frame_prev = cv2.resize(frame_prev, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        if frame_next is not None:
            frame_next = cv2.resize(frame_next, self.size[::-1], interpolation=cv2.INTER_LINEAR)

        if label is not None:
            label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label


# Data augmentation with a random resizing
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class RandScale(object):
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio

        if frame_current is not None:
            frame_current = cv2.resize(frame_current, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        
        if frame_prev is not None:
            frame_prev = cv2.resize(frame_prev, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        
        if frame_next is not None:
            frame_next = cv2.resize(frame_next, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        
        if label is not None:
            label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
            
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label


# Data augmentation with random cropping
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        if label is not None:
            h, w = label.shape
        elif frame_current is not None:
            h, w = frame_current.shape[0], frame_current.shape[1]
        elif frame_prev is not None:
            h, w = frame_prev.shape[0], frame_prev.shape[1]
        else:
            assert False

        assert h > self.crop_h and w > self.crop_w

        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        if frame_current is not None:
            frame_current = frame_current[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if frame_prev is not None:
            frame_prev = frame_prev[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if frame_next is not None:
            frame_next = frame_next[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if label is not None:
            label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]

        if mvs_left is not None and mvs_right is not None:
            mvs_left, mvs_right = crop_motion_vector(mvs_left, mvs_right, h, w, self.crop_h, self.crop_w, h_off, w_off)

        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label

# Crops the grid matrices to the pixels corresponding to the nearest grid structure
def crop_motion_vector(mvs_left, mvs_right, height, width, crop_height, crop_width, height_offset, width_offset):
    if mvs_left is not None and isinstance(mvs_left, list) and len(mvs_left) > 0 and len(mvs_left[0].shape) >= 3:
        motion_vector_height, motion_vector_width = mvs_left[0].shape[-3], mvs_left[0].shape[-2]
    elif mvs_right is not None and isinstance(mvs_right, list) and len(mvs_right) > 0 and len(mvs_left[0].shape) >= 3:
        motion_vector_height, motion_vector_width = mvs_right[0].shape[-3], mvs_right[0].shape[-2]
    else:
        return mvs_left, mvs_right

    pixel_per_block_height = height / motion_vector_height
    pixel_per_block_width = width / motion_vector_width

    final_block_height = crop_height // 16
    final_block_width = crop_width // 16

    # Find the pixels that correspond to the nearest edges in the grid structure
    block_height_offset = round(height_offset / pixel_per_block_height)
    block_width_offset = round(width_offset / pixel_per_block_width)
    block_height = round((height_offset+crop_height) / pixel_per_block_height) - block_height_offset
    block_width = round((width_offset+crop_width) / pixel_per_block_width) - block_width_offset

    # Crop grid matrix, normalize the values the new crop and resize them to the same size
    def _crop_motion_vector(m):
        m = m[block_height_offset:block_height_offset+block_height, block_width_offset:block_width_offset+block_width]
        m[:,:,0] = ((((m[:,:,0] + 1) / 2) * width - width_offset) / (block_width*pixel_per_block_width)) * 2 - 1
        m[:,:,1] = ((((m[:,:,1] + 1) / 2) * height - height_offset) / (block_height*pixel_per_block_height)) * 2 - 1
        # Ensure that all block motion vectors are cropped to the same size
        m = cv2.resize(m, (final_block_width, final_block_height), interpolation=cv2.INTER_LINEAR)
        return m

    # Crop tensor and numpy arrays
    def crop_motion_vector(m):
        if isinstance(m, torch.Tensor):
            device = m.device
            m = m.cpu().numpy()[0]
        else:
            device = None
        m = _crop_motion_vector(m)
        if device is not None:
            m = torch.from_numpy(m).unsqueeze(0).to(device=device)
        return m

    if mvs_left is not None:
        mvs_left = list(map(crop_motion_vector, mvs_left))
    if mvs_right is not None:        
        mvs_right = list(map(crop_motion_vector, mvs_right))

    return mvs_left, mvs_right



# Data augmentation with a random rotation
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            
            if label is not None:
                h, w = label.shape
            elif frame_current is not None:
                h, w = frame_current.shape[0], frame_current.shape[1]
            elif frame_prev is not None:
                h, w = frame_prev.shape[0], frame_prev.shape[1]
            else:
                assert False

            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        
            if frame_current is not None:
                frame_current = cv2.warpAffine(frame_current, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            
            if frame_prev is not None:
                frame_prev = cv2.warpAffine(frame_prev, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)

            if frame_next is not None:
                frame_next = cv2.warpAffine(frame_next, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)

            if label is not None:
                label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label


# Data augmentation with randomized horizontal flipping
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        if random.random() < self.p:
            if frame_current is not None:
                frame_current = cv2.flip(frame_current, 1)
            if frame_prev is not None:
                frame_prev = cv2.flip(frame_prev, 1)
            if frame_next is not None:
                frame_next = cv2.flip(frame_next, 1)

            def flip_motion_vector(m):
                m = cv2.flip(m, 1)
                m[:,:,0] = m[:,:,0] * -1
                return m

            if mvs_left is not None:
                mvs_left = list(map(flip_motion_vector, mvs_left))
            if mvs_right is not None:
                mvs_right = list(map(flip_motion_vector, mvs_right))

            if label is not None:
                label = cv2.flip(label, 1)
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label


# Data augmentation with a random gaussian blur
# Source: https://github.com/hszhao/semseg/blob/master/util/transform.py
class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        if random.random() < 0.5:
            if frame_current is not None:
                frame_current = cv2.GaussianBlur(frame_current, (self.radius, self.radius), 0)
            if frame_prev is not None:
                frame_prev = cv2.GaussianBlur(frame_prev, (self.radius, self.radius), 0)
            if frame_next is not None:
                frame_next = cv2.GaussianBlur(frame_next, (self.radius, self.radius), 0)
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label

# Ignore a set of classes in the labels and project them to the Background class
class IgnoreClasses(object):
    def __init__(self, classes_to_ignore=None):
        self.classes_to_ignore = classes_to_ignore
        if not self.classes_to_ignore:
            self.classes_to_ignore = []

    def __call__(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, label, *args, **kwargs):
        if label is not None:
            for c in self.classes_to_ignore:
                label[label == c] = 0
        return frame_current, frame_prev, frame_next, mvs_left, mvs_right, label
    