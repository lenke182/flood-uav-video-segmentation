# Implements common code for a frame interpolation model across training implementations
# Implementation is based on Jain and Gonzalez (https://arxiv.org/pdf/1803.07742.pdf)
# Sources:
# - https://github.com/hszhao/semseg/blob/master/tool/train.py
# - https://github.com/hszhao/semseg/blob/master/tool/test.py
# - https://github.com/hszhao/semseg/blob/master/tool/demo.py

import numpy as np
import torch
import wandb
import os
import imageio
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from torch.utils.data import DataLoader

from model.pspnet import FlowPSPNet
from model.deeplabv3 import FlowDeepLabv3

from base.foundation import BaseModel, BaseDataModule, mean, std, is_cpu, round_train

from flow.model import FlowModel
import flow.transform
from util.util import colorize, AverageMeter
from flow.transform import crop_motion_vector

# Hyperparameter for optimizer
@dataclass_json
@dataclass
class OptimKWArgs(dict):
    optim: str = "SGD"
    lr: float = 0.0001
    lr_D: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.0001

# Hyperparameter for loss function
@dataclass_json
@dataclass
class LossKWArgs(dict):
    loss: str = "cross_entropy"
    thresh: float = 0.7
    min_kept: int = 100000


# Base model for the segmentation task with frame interpolation
class FlowBaseModel(BaseModel):

    def __init__(self, 
                test_h: int = 873,
                test_w: int = 873,
                arch: str = "pspnet",
                aux: bool = False,
                loss: LossKWArgs = LossKWArgs(),
                optim: OptimKWArgs = OptimKWArgs(),
                semisupervised: bool = False,
                threshold_st = 0.6,
                lambda_fm = 0.1,
                lambda_st = 1.0,
                feature_based: bool = True,
                no_warp: bool = False,
                no_interpolation_percentage: float = 0.0,
                layers: int = 101,
                zoom_factor: int = 8,
                compute_metrics = True,
                save_images = False,
                save_video = True,
                *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure parameters are correctly logged by Weights and Biases
        loss = loss.to_dict()
        optim = optim.to_dict()
        test_w = round_train(test_w, arch)
        test_h = round_train(test_h, arch)
        self.save_hyperparameters()
        self.hparams.test_w = round_train(test_w, arch)
        self.hparams.test_h = round_train(test_h, arch)
        self.hparams.loss = LossKWArgs.from_dict(loss)
        self.hparams.optim = OptimKWArgs.from_dict(optim)
        
    # Create and return a new instance of the generator
    def get_new_model_arch_G(self):
        # Initialize generator
        arch = self.hparams.arch

        self.hparams.test_w = round_train(self.hparams.test_w, self.hparams.arch)
        self.hparams.test_h = round_train(self.hparams.test_h, self.hparams.arch)

        if arch == "pspnet":
            model = FlowPSPNet(hparams=self.hparams)
            modules_head = [model.decoder, model.ppm]
            modules_back = [model.layers]
        elif arch == 'deeplabv3':
            model = FlowDeepLabv3(hparams=self.hparams)
            modules_head = [model.decoder]
            modules_back = [model.encoder]
        else:
            return None

        # Wrap the underlying neural network in a frame interpolation model
        model = FlowModel(model, feature_based=self.hparams.feature_based, no_warp=self.hparams.no_warp, no_interpolation_percentage=self.hparams.no_interpolation_percentage)

        return model, modules_head, modules_back

    # Initialize all models, sync batchnorm and ensure classification layers are trained with a higher learning rate
    def init_model(self):
        base_lr = self.hparams.optim.lr

        self.model_G, modules_head, modules_back = self.get_new_model_arch_G()
        self.model_G_params_list = []
        for m in modules_back:
            self.model_G_params_list.append({"params": m.parameters(), "lr": base_lr})

        for m in modules_head:
            self.model_G_params_list.append({"params": m.parameters(), "lr": base_lr*10})

        self.param_groups_index = len(modules_back)

        if self.hparams.semisupervised:
            self.model_D = self.get_new_model_arch_D()
            
        
        if not is_cpu():
            self.model_G = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_G)
            if self.hparams.semisupervised:
                self.model_D = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_D)

    # Predicts an output during inference
    def forward(self, frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index):
        return self.model_G(None, frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index)

    def training_step(self, batch, batch_idx):
        assert False

    # Single processing step for the validation set
    def validation_step(self, batch, batch_idx):
        frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index, label = batch["frame_prev"], batch["frame_next"], batch["mvs_left"], batch["mvs_right"], batch["left_index"], batch["right_index"], batch["label"]

        outs = self.forward(frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index)

        output = outs["pred"]
        output = output.data.max(1)[1]

        intersection, union, target = self.compute_metrics(output, label)
        self.intersection_meter_val.update(intersection), self.union_meter_val.update(union), self.target_meter_val.update(target)

    # Single processing step for the test set
    # Segmentation maps are computed by cropping different parts of the image 
    # and stitching the outputs together for the segmentation map for the entire image
    # OR by directly computing the output without cropping and stitching
    def test_step(self, batch, batch_idx):    
        batch_test, test_idx = batch
        frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index, label = batch_test["frame_prev"], batch_test["frame_next"], batch_test["mvs_left"], batch_test["mvs_right"], batch_test["left_index"], batch_test["right_index"], batch_test["label"]

        assert frame_prev.shape[0] == 1 and label.shape[0] == 1

        with self.trainer.profiler.profile("test_interference"):
            if self.hparams.no_cropping:
                output = self.forward(frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index)["pred"]
            else:
                output = self.compute_output(frame_prev.shape[0], self.compute_test_crop, frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index)
            output = output.data.max(1)[1]

        intersection, union, target = self.compute_metrics(output, label)
        
        if test_idx > 0:
            # If Texas video
            self.intersection_meter_test2.update(intersection), self.union_meter_test2.update(union), self.target_meter_test2.update(target)
        else:               
            # If Florida video
            self.intersection_meter_test1.update(intersection), self.union_meter_test1.update(union), self.target_meter_test1.update(target)


    # Computes output for an image by cropping different parts of the image and stitching
    # the results together 
    # Source: https://github.com/hszhao/semseg/blob/master/tool/test.py
    def compute_output(self, n, function, frame_prev, frame_next, mvs_left, mvs_right, *args, **kwargs):
        stride_rate= 2/3
        crop_h, crop_w = self.hparams.test_h, self.hparams.test_w
        _, _, new_h, new_w = frame_prev.shape
        stride_h = int(np.ceil(crop_h * stride_rate))
        stride_w = int(np.ceil(crop_w * stride_rate))
        grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
        prediction_crop = torch.zeros((n, self.hparams.classes, new_h, new_w), dtype=float, device=frame_prev.device)
        count_crop = torch.zeros((new_h, new_w), dtype=float, device=frame_prev.device)
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + crop_h, new_h)
                s_h = e_h - crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + crop_w, new_w)
                s_w = e_w - crop_w
                frame_prev_crop = frame_prev[:, :, s_h:e_h, s_w:e_w].clone()
                frame_next_crop = frame_next[:, :, s_h:e_h, s_w:e_w].clone()
                
                # Crops the grid matrices based on the crop_motion_vector() method
                mvs_left_crop, mvs_right_crop = crop_motion_vector(mvs_left, mvs_right, new_h, new_w, e_h-s_h, e_w-s_w, s_h, s_w)

                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[:, :, s_h:e_h, s_w:e_w] += function(frame_prev_crop, frame_next_crop, mvs_left_crop, mvs_right_crop, *args, **kwargs)
        prediction_crop /= count_crop.unsqueeze(0).unsqueeze(0)
        return prediction_crop

    # Predict the output for a crop during testing
    # Source: https://github.com/hszhao/semseg/blob/master/tool/test.py
    def compute_test_crop(self, frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index):        
        output = self.forward(frame_prev, frame_next, mvs_left, mvs_right, left_index, right_index)["pred"]

        _, _, h_i, w_i = frame_prev.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        
        return output

    # Predict the output for a crop during inference
    # Source: https://github.com/hszhao/semseg/blob/master/tool/test.py
    def compute_predict_crop(self, frame_prev, *args, **kwargs):
        output = self.model_G.predict(frame_prev, *args, **kwargs)["pred"]

        _, _, h_i, w_i = frame_prev.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        return output

    def on_predict_start(self):
        # Init predict metrics
        self.intersection_meter_predict = AverageMeter()
        self.union_meter_predict = AverageMeter()
        self.target_meter_predict = AverageMeter()

        # Load colors for classes
        colors_path = os.path.join(self.hparams.data_root, "list", "colors.txt")
        self.colors = np.loadtxt(colors_path).astype('uint8')

        # Last segmentation map during predict for temporal consistency metric
        self.last_output = None

        # Initialize video writer
        if self.hparams.save_video:
            os.makedirs(os.path.join(self.logger.log_dir, "video"), exist_ok=True)
            video_filename = os.path.join(self.logger.log_dir, "video", self.hparams.predict_v_id + ".avi")
            self.video_writer = imageio.get_writer(video_filename, fps=25, mode="I", format='FFMPEG', macro_block_size=None)
        else:
            self.video_writer = None


    # Single inference step for the frames between two key frames
    def predict_step(self, batch, batch_idx):
        batch_test = batch
        frame_prev, frame_next, mvs_left, mvs_right, frame_id = batch_test["frame_prev"], batch_test["frame_next"], batch_test["mvs_left"], batch_test["mvs_right"], batch_test["frame_id"]
        
        assert frame_prev.shape[0] == 1
        assert len(mvs_left) == len(mvs_right)

        n = len(mvs_left)+1

        # Predict the output of all frames between the two key frames
        with self.trainer.profiler.profile("predict_interference"):
            if self.hparams.no_cropping:
                output = self.model_G.predict(frame_prev, frame_next, mvs_left, mvs_right, n, self.trainer.profiler)["pred"]
            else:
                output = self.compute_output(n, self.compute_predict_crop, frame_prev, frame_next, mvs_left, mvs_right, n, self.trainer.profiler)
            
            output = F.interpolate(output, (1072, 1920), mode='bilinear', align_corners=True)
            output = output.data.max(1)[1]
            output_numpy = output.data.cpu().numpy().astype('uint8')
        
        # Temporal consistency between frames
        if self.hparams.compute_metrics:
            for p in range(n):
                if p > 0:
                    current_output = output[p]
                    next_output = output[p-1]
                elif self.last_output is not None:
                    current_output = output[p]
                    next_output = self.last_output
                else:
                    current_output = None
                    next_output = None

                if current_output is not None and next_output is not None:
                    intersection, union, target = self.compute_metrics(current_output.unsqueeze(0), next_output.unsqueeze(0))
                    self.intersection_meter_predict.update(intersection), self.union_meter_predict.update(union), self.target_meter_predict.update(target)
            self.last_output = output[n-1]

        # Save segmentation maps into individual images
        if self.hparams.save_images:
            os.makedirs(os.path.join(self.logger.log_dir, "frames", self.hparams.predict_v_id), exist_ok=True)
            frame_id = int(frame_id[0].detach().cpu())

            for p in range(n):
                # Save output to dir
                output_color = colorize(np.uint8(output_numpy[p]), self.colors)
                output_color.save(os.path.join(self.logger.log_dir, "frames", self.hparams.predict_v_id, str(frame_id + p) + ".png"))      
              
        # Save segmentation maps into video file
        if self.hparams.save_video:
            for p in range(n):
                frame = self.colors[output_numpy[p]]

                self.video_writer.append_data(frame)

    # Saves the metrics from inference to Weights and Biases
    # and closes the video writer
    def on_predict_end(self):
        if self.video_writer is not None:
            self.video_writer.close()
        

        if not isinstance(self.trainer.profiler, pl.profilers.PyTorchProfiler):
            d = self.trainer.profiler.recorded_durations["predict_interference"]
            d_mean = np.mean(d)
            d_sum = np.sum(d)

            if wandb.run is not None:
                wandb.summary["predict_time_mean"] = d_mean
                wandb.summary["predict_time_sum"] = d_sum

        if self.intersection_meter_predict.count > 0:
            # Source: https://github.com/hszhao/semseg/blob/master/tool/train.py
            iou_class = self.intersection_meter_predict.sum / (self.union_meter_predict.sum + 1e-10)
            accuracy_class = self.intersection_meter_predict.sum / (self.target_meter_predict.sum + 1e-10)
            accuracy = sum(self.intersection_meter_predict.sum) / (sum(self.target_meter_predict.sum) + 1e-10)
            miou = np.mean(iou_class)
            macc = np.mean(accuracy_class)
            
            if wandb.run is not None:
                wandb.summary["predict_miou1_epoch"] = miou
                wandb.summary["predict_macc1_epoch"] = macc
                wandb.summary["predict_accuracy1_epoch"] = accuracy
                wandb.summary["predict_miou1_epoch_classes"] = iou_class
                wandb.summary["predict_macc1_epoch_classes"] = accuracy_class


# Data module for a frame interpolation model
class FlowBaseDataModule(BaseDataModule):

    def __init__(self, 
                frame_delta: int = 25,
                resize_factor = 1.0,
                resize_factor_test = 1.0,
                resize_factor_predict = 1.0,
                no_random_frame_delta=False,
                *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        default_image_size = (1072, 1920)

        if self.hparams.resize_factor != 1.0:
            self.hparams.scale_min = 1.0
        
        if self.hparams.no_cropping:
            # Enables data augmentation even if frame is not cropped during inference
            no_cropping_factor = 1.5
            resize_size = (int(self.hparams.train_h * no_cropping_factor) + 1, int(self.hparams.train_w * no_cropping_factor) + 1) 
            resize_val_test = (self.hparams.train_h, self.hparams.train_w)
            self.hparams.scale_min = 1.0/no_cropping_factor + 0.001
            self.hparams.scale_max = 1.0
        else:
            resize_size = (int(default_image_size[0]*resize_factor), int(default_image_size[1]*resize_factor))
            resize_val_test = resize_size
        
        # Data augmentation for training set
        transform_train_list = [
            flow.transform.Resize(resize_size), # (h,w)
            flow.transform.IgnoreClasses(self.hparams.data_classes_ignore),
        ]
        transform_train_list.append(flow.transform.RandScale([self.hparams.scale_min, self.hparams.scale_max]))

        # Grid matrices can't be rotated
        if self.hparams.no_warp:
            transform_train_list.append(flow.transform.RandRotate([-10, 10], padding=mean, ignore_label=self.hparams.ignore_index))

        transform_train_list.append(flow.transform.RandomGaussianBlur())
        transform_train_list.append(flow.transform.RandomHorizontalFlip())
        transform_train_list.append(flow.transform.Crop([self.hparams.train_h, self.hparams.train_w], crop_type='rand',
                                ignore_label=self.hparams.ignore_index))
        transform_train_list.append(flow.transform.ToTensor())
        transform_train_list.append(flow.transform.Normalize(mean=mean, std=std))

        self.transform_train = flow.transform.Compose(transform_train_list)


        # Data augmentation for validation set
        transform_val_list = [
            flow.transform.Resize(resize_val_test), # (h,w)
            flow.transform.IgnoreClasses(self.hparams.data_classes_ignore),
        ]
        if not self.hparams.no_cropping:
            transform_val_list.append(flow.transform.Crop([self.hparams.train_h, self.hparams.train_w], crop_type='center',
                           ignore_label=self.hparams.ignore_index))
        transform_val_list.append(flow.transform.ToTensor())
        transform_val_list.append(flow.transform.Normalize(mean=mean, std=std))

        self.transform_val = flow.transform.Compose(transform_val_list)


        # Data augmentation for testing set
        self.transform_test1 = flow.transform.Compose([
            flow.transform.Resize((int(resize_val_test[0]*resize_factor_test), int(resize_size[1]*resize_factor_test))), # (h,w)
            flow.transform.IgnoreClasses(self.hparams.data_classes_ignore),
            flow.transform.ToTensor(),
            flow.transform.Normalize(mean=mean, std=std)
        ])
        self.transform_test2 = flow.transform.Compose([
            flow.transform.Resize((int(resize_val_test[0]*resize_factor_test), int(resize_size[1]*resize_factor_test))), # (h,w)
            flow.transform.IgnoreClasses(self.hparams.data_classes_ignore),
            flow.transform.ToTensor(),
            flow.transform.Normalize(mean=mean, std=std)
        ])

        # Data augmentation during inference
        self.transform_predict = flow.transform.Compose([
            flow.transform.Resize((int(resize_val_test[0]*resize_factor_predict), int(resize_size[1]*resize_factor_predict))), # (h,w)
            flow.transform.IgnoreClasses(self.hparams.data_classes_ignore),
            flow.transform.ToTensor(),
            flow.transform.Normalize(mean=mean, std=std)
        ])

    # DataLoader during inference
    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.hparams.batch_size_test, shuffle=False,
                        num_workers=self.hparams.workers_test, pin_memory=True)