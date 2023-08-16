# Implements the frame interpolation techniques
# Implementation is based on Jain and Gonzalez (https://arxiv.org/pdf/1803.07742.pdf)

import torch
from torch import nn
import numpy as np
import random

# Returns a grid matrix that represents the identity function
def get_default_grid():
    width, height = 1920, 1072
    block_size = 16
    block_height = height // block_size
    block_width = width // block_size
    default_grid = np.zeros((block_height, block_width, 2))
    x = np.linspace(0, block_width-1, block_width)
    y = np.linspace(0, block_height-1, block_height)
    xv, yv = np.meshgrid(x, y)
    default_grid[:,:,0] = (xv * block_size + block_size//2) / width * 2 - 1
    default_grid[:,:,1] = (yv * block_size + block_size//2) / height * 2 - 1
    return default_grid

class FlowModel(nn.Module):
    def __init__(self, model, feature_based=True, no_warp=False, no_interpolation_percentage=0.0):
        super(FlowModel, self).__init__()
        self.model = model
        self.feature_based = feature_based
        self.no_warp = no_warp
        self.no_interpolation_percentage = no_interpolation_percentage

        #Default grid - no motion
        self.default_motion_vector = torch.from_numpy(get_default_grid()).float().unsqueeze(0)

    # Returns an interpolated frame based on two key frames and grid matrices
    def forward(self, frame_current, frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index):
        # Train model without frame interpolation
        if self.training and frame_current is not None and random.random() < self.no_interpolation_percentage:
            h, w = frame_current.shape[2], frame_current.shape[3]
            f = self.model.encoder(frame_current)
            output = self.model.decoder(f)
            if output.shape[2] != h or output.shape[3] != w:
                output = torch.nn.functional.interpolate(output, size=(h,w), mode='bilinear', align_corners=True)
            return {"pred": output}
        else:
            # Train model with frame interpolation
            left_index = [int(i) for i in left_index]
            right_index = [int(i) for i in right_index]
            n_list = [sum(x) for x in zip(left_index, right_index)]
            if self.feature_based:
                return self.forward_feature(frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index, n_list)
            else:
                return self.forward_segmentation(frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index, n_list)

    # Implements training step for feature-based frame interpolation
    def forward_feature(self, frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index, n_list):
        h, w = frame_prev.shape[2], frame_prev.shape[3]
        f_prev = self.model.encoder(frame_prev)
        f_next = self.model.encoder(frame_next)
        f_h, f_w = f_prev.shape[2], f_prev.shape[3]

        f_prev_warped = self.warp_batch(f_prev, mvs_left, left_index, n_list)
        f_next_warped = self.warp_batch(f_next, mvs_right, right_index, n_list)

        f = f_prev_warped + f_next_warped

        output = self.model.decoder(f)
        if output.shape[2] != h or output.shape[3] != w:
            output = torch.nn.functional.interpolate(output, size=(h,w), mode='bilinear', align_corners=True)

        return {"pred": output}
    
    # Implements training step for segmentation-based frame interpolation
    def forward_segmentation(self, frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index, n_list):
        h, w = frame_prev.shape[2], frame_prev.shape[3]

        f_prev = self.model.encoder(frame_prev)
        f_next = self.model.encoder(frame_next)
        o_prev = self.model.decoder(f_prev)
        o_next = self.model.decoder(f_next)

        o_prev_warped = self.warp_batch(o_prev, mvs_left, left_index, n_list)
        o_next_warped = self.warp_batch(o_next, mvs_right, right_index, n_list)

        o = o_prev_warped + o_next_warped
        if o.shape[2] != h or o.shape[3] != w:
            o = torch.nn.functional.interpolate(o, size=(h,w), mode='bilinear', align_corners=True)

        return {"pred": o}
    
    # Warps a batch of feature maps/segmentation maps according to the batch specific frame distance to the key frames
    # and weights them
    def warp_batch(self, input, mvs, index_list, n_list):
        i_h, i_w = input.shape[2], input.shape[3]
        input_warped = []
        for i in range(len(index_list)):
            index = index_list[i]
            input_warped_batch = input[i].unsqueeze(0)
            if not self.no_warp:
                for j in range(index):
                    m = mvs[j][i].unsqueeze(0)
                    input_warped_batch = self.warp(input_warped_batch, m)
                if input_warped_batch.shape[1] != i_h or input_warped_batch.shape[2] != i_w:
                    input_warped_batch = torch.nn.functional.interpolate(input_warped_batch, size=(i_h,i_w), mode='bilinear', align_corners=True)
            input_warped_batch = input_warped_batch * ((n_list[i] - index) / n_list[i])
            input_warped.append(input_warped_batch)
        return torch.cat(input_warped)

    # Run the inference pipeline of the frame interpolation model
    def predict(self, *args, **kwargs):
        if self.feature_based:
            return self.predict_feature(*args, **kwargs)
        else:
            return self.predict_segmentation(*args, **kwargs)

    # Predict the output for a feature-based frame interpolation model
    def predict_feature(self, frame_prev, frame_next, mvs_left, mvs_right, n, profiler):
        h, w = frame_prev.shape[2], frame_prev.shape[3]

        with profiler.profile("predict_encoder"):
            f = self.model.encoder(frame_prev)
        f_h, f_w = f.shape[2], f.shape[3]

        feature_maps = []
        feature_maps_forward = []
        feature_maps_backward = []

        if frame_next is not None:
            with profiler.profile("predict_encoder"):
                f_next = self.model.encoder(frame_next)

            # Warp feature maps forward
            if not self.no_warp:
                f_prev_warped = f
                with profiler.profile("predict_warp"):
                    for m in mvs_left:
                        f_prev_warped = self.warp(f_prev_warped, m)
                        f_prev_warped_inter = f_prev_warped
                        if f_prev_warped_inter.shape[2] != f_h or f_prev_warped_inter.shape[3] != f_w:
                            f_prev_warped_inter = torch.nn.functional.interpolate(f_prev_warped_inter, size=(f_h,f_w), mode='bilinear', align_corners=True)
                        feature_maps_forward.append(f_prev_warped_inter)

            # Warp feature maps backward
            if not self.no_warp:
                f_next_warped = f_next
                with profiler.profile("predict_warp"):
                    for m in mvs_right:
                        f_next_warped = self.warp(f_next_warped, m)
                        f_next_warped_inter = f_next_warped
                        if f_next_warped_inter.shape[2] != f_h or f_next_warped_inter.shape[3] != f_w:
                            f_next_warped_inter = torch.nn.functional.interpolate(f_next_warped_inter, size=(f_h,f_w), mode='bilinear', align_corners=True)
                        feature_maps_backward.append(f_next_warped_inter)

        # Interpolate feature map of key frame with default block motion vector
        if not self.no_warp:
            if self.default_motion_vector.device != f.device:
                self.default_motion_vector = self.default_motion_vector.to(device=f.device)
            f = torch.nn.functional.grid_sample(f, self.default_motion_vector, padding_mode="border", align_corners=True)
            if f.shape[2] != f_h or f.shape[3] != f_w:
                f = torch.nn.functional.interpolate(f, size=(f_h,f_w), mode='bilinear', align_corners=True)
        
        feature_maps.append(f)

        # Fuse feature maps together
        if frame_next is not None:
            with profiler.profile("predict_fusion"):
                for p in range(1,n):
                    if not self.no_warp:
                        f_p = (n-p)/n * feature_maps_forward[p-1] + p/n * feature_maps_backward[n-p-1]
                    else:
                        f_p = (n-p)/n * f + p/n * f_next
                    feature_maps.append(f_p)

        feature_maps = torch.cat(feature_maps, 0)

        # Compute output of all interpolated feature maps in a single execution
        with profiler.profile("predict_decoder"):
            output = self.model.decoder(feature_maps)
            if output.shape[2] != h or output.shape[3] != w:
                output = torch.nn.functional.interpolate(output, size=(h,w), mode='bilinear', align_corners=True)

        return {"pred": output}

    # Predict the output for a segmentation-based frame interpolation model
    def predict_segmentation(self, frame_prev, frame_next, mvs_left, mvs_right, n, profiler):
        h, w = frame_prev.shape[2], frame_prev.shape[3]

        # Predict segmentation map at key frame
        with profiler.profile("predict_encoder"):
            f = self.model.encoder(frame_prev)
        with profiler.profile("predict_decoder"):
            o = self.model.decoder(f)
            if o.shape[2] != h or o.shape[3] != w:
                o = torch.nn.functional.interpolate(o, size=(h,w), mode='bilinear', align_corners=True)

        segmentation_maps = [
            o
        ]

        if frame_next is not None:
            # Predict segmentation map at key frame
            with profiler.profile("predict_encoder"):
                f_next = self.model.encoder(frame_next)
            with profiler.profile("predict_decoder"):
                o_next = self.model.decoder(f_next)
                if o_next.shape[2] != h or o_next.shape[3] != w:
                    o_next = torch.nn.functional.interpolate(o_next, size=(h,w), mode='bilinear', align_corners=True)

            segmentation_maps_forward = []
            segmentation_maps_backward = []

            # Warp segmentation maps forward
            o_prev_warped = o
            with profiler.profile("predict_warp"):
                for m in mvs_left:
                    o_prev_warped = self.warp(o_prev_warped, m)
                    o_prev_warped_inter = o_prev_warped
                    if o_prev_warped_inter.shape[2] != h or o_prev_warped_inter.shape[3] != w:
                        o_prev_warped_inter = torch.nn.functional.interpolate(o_prev_warped_inter, size=(h,w), mode='bilinear', align_corners=True)
                    segmentation_maps_forward.append(o_prev_warped_inter)

            # Warp segmentation maps backward
            o_next_warped = o_next
            with profiler.profile("predict_warp"):
                for m in mvs_right:
                    o_next_warped = self.warp(o_next_warped, m)
                    o_next_warped_inter = o_next_warped
                    if o_next_warped_inter.shape[2] != h or o_next_warped_inter.shape[3] != w:
                        o_next_warped_inter = torch.nn.functional.interpolate(o_next_warped_inter, size=(h,w), mode='bilinear', align_corners=True)
                    segmentation_maps_backward.append(o_next_warped_inter)

            # Fuse segmentation maps together
            with profiler.profile("predict_fusion"):
                for p in range(1,n):
                    o_prev_warped = (n-p)/n * segmentation_maps_forward[p-1]
                    o_next_warped = p/n * segmentation_maps_backward[n-p-1]
                    o_p = o_prev_warped + o_next_warped
                    segmentation_maps.append(o_p)

        segmentation_maps = torch.cat(segmentation_maps, 0)

        return {"pred": segmentation_maps}

    # Warp a feature or segmentation map using a grid matrix
    def warp(self, frame, motion_vectors):
        if not self.no_warp:
            if not isinstance(motion_vectors, torch.FloatTensor):
                motion_vectors = motion_vectors.float()
            frame = torch.nn.functional.grid_sample(frame, motion_vectors, mode="bilinear", padding_mode="border", align_corners=False)
        return frame