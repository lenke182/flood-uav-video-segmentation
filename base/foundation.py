# Implements the base functionality used by all training methods and segmentation tasks

# Sources:
# - https://github.com/hszhao/semseg/blob/master/tool/train.py
# - https://github.com/hszhao/semseg/blob/master/tool/test.py
# - https://github.com/hszhao/semseg/blob/master/tool/demo.py

import cv2
import os
import wandb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from util.util import poly_learning_rate, intersectionAndUnionGPU, intersectionAndUnion, AverageMeter
from torch.utils.data import DataLoader
from util import dataset, transform
import bisect

LOG_BATCHES = 10

def is_cpu():
    return torch.cuda.device_count() == 0

# Mean and standard deviation for normalizing data
# Source: https://github.com/hszhao/semseg/blob/master/tool/train.py
value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

# Round crop size to the nearest permitted crop size
def round_train(train, arch):
    if arch == "pspnet":
        return (train-1)//8*8+1
    elif arch == "vit":
        return train//32*32
    elif arch == 'deeplabv3':
        return (train-1)//8*8+1
    else:
        assert False


# Implements common functionality across all training methods and frame interpolation methods
class BaseModel(pl.LightningModule):
    def __init__(self,
                 classes: int = 5,
                 layers: int = 101,
                 test_h: int = 873,
                 test_w: int = 873,
                 arch: str = "pspnet",
                 ignore_index = 255,
                 data_root: str = "dataset/flow/",
                 test_scales=[1.0],
                 power=0.9,
                 no_cropping = False,
                 no_warp = False,
                 semisupervised = False,
                 predict_v_id = "florida-01",
                 only_segmentation = False,
                 pretrained=True,
                 *args, **kwargs):
        super().__init__()

        # Ensure parameters are correctly logged by Weights and Biases
        test_w = round_train(test_w, arch)
        test_h = round_train(test_h, arch)
        self.save_hyperparameters()
        self.hparams.test_w = round_train(test_w, arch)
        self.hparams.test_h = round_train(test_h, arch)

        # Disable automatic optimization by PyTorch Lightning
        self.automatic_optimization = False

        torch.use_deterministic_algorithms(True)
        
        # Initialize base model
        self.init_loss_function()
        self.init_model()       
        self.init_metrics_train()
        self.init_metrics_val()
        self.init_metrics_test()


    # Initialize metrics for training set 
    def init_metrics_train(self):
        self.intersection_meter_train = AverageMeter()
        self.union_meter_train = AverageMeter()
        self.target_meter_train = AverageMeter()

    # Initialize metrics for validation set
    def init_metrics_val(self):
        self.intersection_meter_val = AverageMeter()
        self.union_meter_val = AverageMeter()
        self.target_meter_val = AverageMeter()

    # Initialize metrics for test sets
    def init_metrics_test(self):
        self.intersection_meter_test1 = AverageMeter()
        self.union_meter_test1 = AverageMeter()
        self.target_meter_test1 = AverageMeter()
        self.intersection_meter_test2 = AverageMeter()
        self.union_meter_test2 = AverageMeter()
        self.target_meter_test2 = AverageMeter()
        self.batches_logged = 0

    def init_loss_function(self):
        assert False

    def init_model(self):
        assert False

    def forward(self, x):
        assert False

    # Returns the current global training step
    def get_total_batch_idx(self, batch_idx):
        return self.trainer.estimated_stepping_batches/self.trainer.max_epochs * self.trainer.current_epoch + batch_idx

    # Sets the learning rate for parameter groups
    # Classification layers are trained with a higher base learning rate than the pre-trained layers
    # Source: https://github.com/hszhao/semseg/blob/master/tool/train.py
    def learning_rate_step(self, opt, base_lr, batch_idx):
        total_batch_idx = self.get_total_batch_idx(batch_idx) + 1
        current_lr = poly_learning_rate(base_lr, total_batch_idx, self.trainer.estimated_stepping_batches, power=self.hparams.power)
        self.log("train_lr", current_lr, on_step=True, on_epoch=True, sync_dist=True)
        for i in range(0, len(opt.param_groups)):
            if i < self.param_groups_index:
                opt.param_groups[i]['lr'] = current_lr
            else:
                opt.param_groups[i]['lr'] = current_lr * 10
    
    # Log metrics from training epoch locally and to Weights and Biases
    def training_epoch_end(self, outputs) -> None:
        # Source: https://github.com/hszhao/semseg/blob/master/tool/train.py
        miou = np.mean(self.intersection_meter_train.sum / (self.union_meter_train.sum + 1e-10))
        macc = np.mean(self.intersection_meter_train.sum / (self.target_meter_train.sum + 1e-10))
        accuracy = sum(self.intersection_meter_train.sum) / (sum(self.target_meter_train.sum) + 1e-10)

        self.log("train_miou_epoch", miou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_macc_epoch", macc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_accuracy_epoch", accuracy, on_step=False, on_epoch=True, sync_dist=True)

        self.init_metrics_train()

        return super().training_epoch_end(outputs)

    # Single processing step for the validation set
    def validation_step(self, batch, batch_idx):    
        images, labels = batch

        output = self.forward(images)["pred"]
        output = output.data.max(1)[1]

        intersection, union, target = self.compute_metrics(output, labels)
        self.intersection_meter_val.update(intersection), self.union_meter_val.update(union), self.target_meter_val.update(target)

    # Computes and saves metrics from validation set
    def validation_epoch_end(self, outputs) -> None:
        # Source: https://github.com/hszhao/semseg/blob/master/tool/train.py
        miou = np.mean(self.intersection_meter_val.sum / (self.union_meter_val.sum + 1e-10))
        macc = np.mean(self.intersection_meter_val.sum / (self.target_meter_val.sum + 1e-10))
        accuracy = sum(self.intersection_meter_val.sum) / (sum(self.target_meter_val.sum) + 1e-10)

        self.log("val_miou_epoch", miou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_macc_epoch", macc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy_epoch", accuracy, on_step=False, on_epoch=True, sync_dist=True)

        self.init_metrics_val()

        return super().validation_epoch_end(outputs)

    # Single processing step for a image from the test set
    # Segmentation maps are computed by cropping different parts of the image 
    # and stitching the outputs together for the segmentation map for the entire image
    def test_step(self, batch, batch_idx):
        batch_test, test_idx = batch
        input, target = batch_test

        assert input.shape[0] == 1 and target.shape[0] == 1

        base_size = 2048

        # Process image at different scales
        with self.trainer.profiler.profile("test_interference"):
            # Source: https://github.com/hszhao/semseg/blob/master/tool/test.py
            input = np.squeeze(input.cpu().numpy(), axis=0)
            image = np.transpose(input, (1, 2, 0))
            h, w, _ = image.shape
            prediction = np.zeros((h, w, self.hparams.classes), dtype=float)
            for scale in self.hparams.test_scales:
                long_size = round(scale * base_size)
                new_h = long_size
                new_w = long_size
                if h > w:
                    new_w = round(long_size / float(h) * w)
                else:
                    new_h = round(long_size / float(w) * h)
                image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                prediction += self.compute_test_output_for_scales(image_scale, h, w)
            prediction /= len(self.hparams.test_scales)
            prediction = np.argmax(prediction, axis=2)

            target = np.squeeze(target.cpu().numpy(), axis=0)

        intersection, union, target = self.compute_metrics(prediction, target)
        
        if test_idx > 0:
            # If Texas video
            self.intersection_meter_test2.update(intersection), self.union_meter_test2.update(union), self.target_meter_test2.update(target)
        else:               
            # If Florida video
            self.intersection_meter_test1.update(intersection), self.union_meter_test1.update(union), self.target_meter_test1.update(target)

        # Log predictions to Weights and Biases
        if self.batches_logged < LOG_BATCHES and test_idx > 0:
            self.batches_logged += 1
            return prediction
        else:
            return None

    # Save metrics of testing set locally and to Weights and Biases
    def test_epoch_end(self, outputs) -> None:
        # Source: https://github.com/hszhao/semseg/blob/master/tool/train.py
        iou_class1 = self.intersection_meter_test1.sum / (self.union_meter_test1.sum + 1e-10)
        accuracy_class1 = self.intersection_meter_test1.sum / (self.target_meter_test1.sum + 1e-10)
        accuracy1 = sum(self.intersection_meter_test1.sum) / (sum(self.target_meter_test1.sum) + 1e-10)
        miou1 = np.mean(iou_class1)
        macc1 = np.mean(accuracy_class1)

        self.log("test_miou1_epoch", miou1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_macc1_epoch", macc1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_accuracy1_epoch", accuracy1, on_step=False, on_epoch=True, sync_dist=True)
        if wandb.run is not None:
            wandb.summary["test_miou1_epoch_classes"] = iou_class1
            wandb.summary["test_macc1_epoch_classes"] = accuracy_class1
        
        if self.intersection_meter_test2.count > 0:
            iou_class2 = self.intersection_meter_test2.sum / (self.union_meter_test2.sum + 1e-10)
            accuracy_class2 = self.intersection_meter_test2.sum / (self.target_meter_test2.sum + 1e-10)
            accuracy2 = sum(self.intersection_meter_test2.sum) / (sum(self.target_meter_test2.sum) + 1e-10)
            miou2 = np.mean(iou_class2)
            macc2 = np.mean(accuracy_class2)

            self.log("test_miou2_epoch", miou2, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("test_macc2_epoch", macc2, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_accuracy2_epoch", accuracy2, on_step=False, on_epoch=True, sync_dist=True)
            if wandb.run is not None:
                wandb.summary["test_miou2_epoch_classes"] = iou_class2
                wandb.summary["test_macc2_epoch_classes"] = accuracy_class2
        
            self.log("test_miou_epoch", (miou1+miou2)/2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_macc_epoch", (macc1+macc2)/2, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_accuracy_epoch", (accuracy1+accuracy2)/2, on_step=False, on_epoch=True, sync_dist=True)

        self.init_metrics_test()

        return super().test_epoch_end(outputs)

    # Computes output for an image by cropping different parts of the image and stitching
    # the results together 
    # Source: https://github.com/hszhao/semseg/blob/master/tool/test.py
    def compute_test_output_for_scales(self, image, h, w, stride_rate=2 / 3):
        crop_h, crop_w = self.hparams.test_h, self.hparams.test_w
        ori_h, ori_w, _ = image.shape
        pad_h = max(crop_h - ori_h, 0)
        pad_w = max(crop_w - ori_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=mean)
        new_h, new_w, _ = image.shape
        stride_h = int(np.ceil(crop_h * stride_rate))
        stride_w = int(np.ceil(crop_w * stride_rate))
        grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
        prediction_crop = np.zeros((new_h, new_w, self.hparams.classes), dtype=float)
        count_crop = np.zeros((new_h, new_w), dtype=float)
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + crop_h, new_h)
                s_h = e_h - crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + crop_w, new_w)
                s_w = e_w - crop_w
                image_crop = image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[s_h:e_h, s_w:e_w, :] += self.net_process(image_crop)
        prediction_crop /= np.expand_dims(count_crop, 2)
        prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
        prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return prediction

    # Returns a segmentation map for a single cropped part
    # Source: https://github.com/hszhao/semseg/blob/master/tool/test.py
    def net_process(self, image, flip=True):
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()
        if std is None:
            for t, m in zip(input, mean):
                t.sub_(m)
        else:
            for t, m, s in zip(input, mean, std):
                t.sub_(m).div_(s)
        input = input.unsqueeze(0)
        if not is_cpu():
            input = input.cuda()
        if flip:
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            output = self.forward(input)
            if not isinstance(output, torch.Tensor):
                output = output["pred"]
            if output.shape[1] != self.hparams.classes:
                output = output[:, 0:self.hparams.classes, :, :]

        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output

    # Computes the metrics on the CPU or GPU
    def compute_metrics(self, output, target, cpu=False):
        if cpu or is_cpu() or (isinstance(output, np.ndarray) and isinstance(target, np.ndarray)):
            if isinstance(output, torch.Tensor):
                output = output.detach().numpy()
            if isinstance(target, torch.Tensor):
                target = target.detach().numpy()
            return intersectionAndUnion(output, target, self.hparams.classes,
                                               self.hparams.ignore_index)
        else:
            intersection, union, target = intersectionAndUnionGPU(output, target, self.hparams.classes,
                                               self.hparams.ignore_index)
            return intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()


# Reimplementation of the PyTorch ConcatDataset class to also return the id of the dataset
# Source: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataset.py
class BetterConcatDataset(torch.utils.data.ConcatDataset):

    def __init__(self, *args, **kwargs) -> None:
        super(BetterConcatDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

# Data module that returns a training, validation and test set
# Source: https://github.com/hszhao/semseg/blob/master/tool/train.py
class BaseDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_root: str = "dataset/flow/",
                 arch: str = "pspnet",
                 data_variant = None,
                 data_classes_ignore = None,
                 batch_size: int = 4,
                 batch_size_val: int = 4,
                 batch_size_test: int = 1,
                 train_h: int = 873,
                 train_w: int = 873,
                 workers: int = 8,
                 workers_test: int = 8,
                 ignore_index: int = 255,
                 semisupervised: bool = False,
                 data_ratio = 1.0,
                 scale_min = 0.5,
                 scale_max = 2.0,
                 no_cropping = False,
                 no_warp = False,
                 predict_v_id = "florida-01",
                 *args, **kwargs):
        super().__init__()

        # Ensures hyperparameter are correctly logged to Weights and Biases
        train_w = round_train(train_w, arch)
        train_h = round_train(train_h, arch)
        self.save_hyperparameters()
        self.hparams.train_w = round_train(train_w, arch)
        self.hparams.train_h = round_train(train_h, arch)

        # Data augmentation for the training set
        self.transform_train = transform.Compose([
            transform.IgnoreClasses(self.hparams.data_classes_ignore),
            transform.Resize((1072, 1920)), # (h,w)
            transform.RandScale([scale_min, scale_max]),
            transform.RandRotate([-10, 10], padding=mean, ignore_label=self.hparams.ignore_index),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([train_h, train_w], crop_type='rand', padding=mean,
                           ignore_label=self.hparams.ignore_index),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ])

        # Data augmentation for the validation set
        self.transform_val = transform.Compose([
            transform.IgnoreClasses(self.hparams.data_classes_ignore),
            transform.Resize((1072, 1920)), # (h,w)
            transform.Crop([train_h, train_w], crop_type='center', padding=mean,
                           ignore_label=self.hparams.ignore_index),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ])

        # Data augmentation for both test videos
        self.transform_test1 = transform.Compose([
            transform.IgnoreClasses(self.hparams.data_classes_ignore),
            transform.Resize((1072, 1920)), # (h,w)
            transform.ToTensor()
        ])
        self.transform_test2 = transform.Compose([
            transform.IgnoreClasses(self.hparams.data_classes_ignore),
            transform.Resize((1072, 1920)), # (h,w)
            transform.ToTensor()
        ])

    # Setup PyTorch Dataset classes
    def setup(self, stage: str):
        if stage == "fit":
            data_train = dataset.SemData(split='train', data_root=self.hparams.data_root,
                                            data_list=self.get_list_path("train.txt"),
                                            transform=self.transform_train)

            # If independent unsupervised dataset doesn't exist, split supervised set into supervised and unsupervised
            data_train_u_path = self.get_list_path("train_u.txt")
            if os.path.exists(data_train_u_path):
                self.dataset_train_l = data_train
                self.dataset_train_u = dataset.SemData(split='test', data_root=self.hparams.data_root,
                                                data_list=data_train_u_path,
                                                transform=self.transform_train)
            else:
                size_l = int(self.hparams.data_ratio * len(data_train))
                size_u = len(data_train) - size_l
                self.dataset_train_l, self.dataset_train_u = torch.utils.data.random_split(data_train, [size_l, size_u])

            self.data_val = dataset.SemData(split='val', data_root=self.hparams.data_root,
                                            data_list=self.get_list_path("val.txt"),
                                            transform=self.transform_val)
        if stage == "test":
            self.data_test1 = dataset.SemData(split='val', data_root=self.hparams.data_root,
                                             data_list=self.get_list_path("test.txt"),
                                             transform=self.transform_test1)
            self.data_test2 = dataset.SemData(split='val', data_root=self.hparams.data_root,
                                             data_list=self.get_list_path("test2.txt"),
                                             transform=self.transform_test2)
            self.data_test = BetterConcatDataset([self.data_test1, self.data_test2])
        if stage == "predict":
            self.data_predict = dataset.SemData(split='val', data_root=self.hparams.data_root,
                                                data_list=self.get_list_path("test.txt"),
                                                transform=self.transform_test)

    # Returns the path for a txt file
    def get_list_path(self, filename):
        if self.hparams.data_variant:
            return os.path.join(self.hparams.data_root, "list", self.hparams.data_variant, filename)
        else:
            return os.path.join(self.hparams.data_root, "list", filename)

    # DataLoader for the training set
    def train_dataloader(self):
        assert False

    # DataLoader for the validation set
    def val_dataloader(self):
        dataloader_val = DataLoader(self.data_val, batch_size=self.hparams.batch_size_val, shuffle=False,
                          num_workers=self.hparams.workers, pin_memory=True)

        return dataloader_val

    # DataLoader for the testing set
    def test_dataloader(self):
        dataloader_test = DataLoader(self.data_test, batch_size=self.hparams.batch_size_test, shuffle=False,
                          num_workers=self.hparams.workers_test, pin_memory=True)

        return dataloader_test
