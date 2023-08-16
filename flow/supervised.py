# Sources:
# - https://github.com/hszhao/semseg/blob/master/tool/train.py

# Implements supervised learning for the segmentation task with frame interpolation

import torch
from torch.utils.data import DataLoader
from base.foundation import BetterConcatDataset

from flow.dataset import FlowData
from flow.base import FlowBaseModel, FlowBaseDataModule
from u2pl.utils.loss_helper import CriterionOhem


# Supervised learning model for frame interpolation
class FlowSupervised(FlowBaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_loss_function(self):
        aux_weight = self.hparams.aux_weight if self.hparams.aux else 0

        # OHEM Cross Entropy Loss
        self.criterion = CriterionOhem(
            aux_weight,
            ignore_index=self.hparams.ignore_index,
            thresh = self.hparams.loss.thresh,
            min_kept = self.hparams.loss.min_kept
        )

    # Supervised learning training step
    def training_step(self, batch, batch_idx):
        batch_l = batch["l"]
        frame_current_l, frame_prev_l, frame_next_l, mvs_left_l, mvs_right_l, left_index_l, right_index_l, label_l = batch_l["frame_current"], batch_l["frame_prev"], batch_l["frame_next"], batch_l["mvs_left"], batch_l["mvs_right"], batch_l["left_index"], batch_l["right_index"], batch_l["label"]

        outs = self.model_G(frame_current_l, frame_prev_l, frame_next_l, mvs_left_l, mvs_right_l, left_index_l, right_index_l)
        pred = outs["pred"]

        loss = self.criterion(pred, label_l)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        pred = pred.max(1)[1]

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.learning_rate_step(opt, self.hparams.optim.lr, batch_idx)

        intersection, union, target = self.compute_metrics(pred, label_l)
        self.intersection_meter_train.update(intersection), self.union_meter_train.update(union), self.target_meter_train.update(target)

    # Initialize the SGD optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model_G_params_list, 
            lr=self.hparams.optim.lr,
            momentum=self.hparams.optim.momentum,
            weight_decay=self.hparams.optim.weight_decay)
    
        return optimizer


# Data module that implements supervised learning for frame interpolation
class FlowSupervisedDataModule(FlowBaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Setups the PyTorch Dataset classes
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset_train_l = FlowData(split='train', type="l", data_root=self.hparams.data_root,
                                            data_list=self.get_list_path("train.txt"),
                                            transform=self.transform_train, frame_delta=self.hparams.frame_delta,
                                            no_warp=self.hparams.no_warp,no_random_frame_delta=self.hparams.no_random_frame_delta)
            
            self.data_val = FlowData(split='val', type="l", data_root=self.hparams.data_root,
                                            data_list=self.get_list_path("val.txt"),
                                            transform=self.transform_val, frame_delta=self.hparams.frame_delta,
                                            no_warp=self.hparams.no_warp)
        if stage == "test":
            self.data_test1 = FlowData(split='test', type="l", data_root=self.hparams.data_root,
                                             data_list=self.get_list_path("test.txt"),
                                             transform=self.transform_test1, frame_delta=self.hparams.frame_delta,
                                             no_warp=self.hparams.no_warp)
            self.data_test2 = FlowData(split='test', type="l", data_root=self.hparams.data_root,
                                             data_list=self.get_list_path("test2.txt"),
                                             transform=self.transform_test2, frame_delta=self.hparams.frame_delta,
                                             no_warp=self.hparams.no_warp)
            self.data_test = BetterConcatDataset([self.data_test1, self.data_test2])
        if stage == "predict":
            self.data_predict = FlowData(split='predict', data_root=self.hparams.data_root,
                                                data_list=self.get_list_path("test.txt"),
                                                transform=self.transform_test2, frame_delta=self.hparams.frame_delta,
                                                no_warp=self.hparams.no_warp)

    # DataLoader for the training set
    def train_dataloader(self):

        dataloader_train_l = DataLoader(self.dataset_train_l, batch_size=self.hparams.batch_size,
                   shuffle=True,
                   num_workers=self.hparams.workers, pin_memory=True,
                   prefetch_factor=1, 
                   drop_last=True)

        return {"l": dataloader_train_l}

    # DataLoader for the validation set
    def val_dataloader(self):
        dataloader_val = DataLoader(self.data_val, batch_size=self.hparams.batch_size_val, shuffle=False,
                          num_workers=self.hparams.workers, prefetch_factor=1, pin_memory=True,
        )

        return dataloader_val
