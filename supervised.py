# Sources:
# - https://github.com/hszhao/semseg/blob/master/tool/train.py

# Implements the supervised learning method for the segmentation task without frame interpolation

from base.cli import FlowLightningCLI

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from model.pspnet import PSPNetSemi
from model.vit import VITSemi
from model.deeplabv3 import DeepLabv3Semi
from u2pl.utils.loss_helper import Criterion, CriterionOhem
from base.foundation import BaseModel, BaseDataModule, is_cpu, round_train

# Hyperparameter related to the loss function
@dataclass_json
@dataclass
class LossKWArgs(dict):
    loss: str = "ohem"
    thresh: float = 0.7
    min_kept: int = 100000

# Hyperparameter related to the optimizer
@dataclass_json
@dataclass
class OptimKWArgs(dict):
    optim: str = "SGD"
    lr: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 0.0001

class SupervisedModel(BaseModel):

    def __init__(self,
                aux: bool = True,
                aux_weight=0.4,
                optim: OptimKWArgs = OptimKWArgs(),
                loss: LossKWArgs = LossKWArgs(),
                test_h: int = 873,
                test_w: int = 873,
                arch: str = "pspnet",
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

    # Initialize the loss functions
    def init_loss_function(self):
        aux_weight = self.hparams.aux_weight if self.hparams.aux else 0

        if self.hparams.loss.loss == "ohem":
            self.criterion = CriterionOhem(
                aux_weight,
                ignore_index=self.hparams.ignore_index,
                thresh = self.hparams.loss.thresh,
                min_kept = self.hparams.loss.min_kept
            )
        else:
            self.criterion = Criterion(
                aux_weight,
                ignore_index=self.hparams.ignore_index,
            )

    # Returns a new segmentation model
    def get_new_model_arch(self):
        arch = self.hparams.arch
        
        self.hparams.test_w = round_train(self.hparams.test_w, self.hparams.arch)
        self.hparams.test_h = round_train(self.hparams.test_h, self.hparams.arch)

        if arch == "pspnet":
            return PSPNetSemi(hparams=self.hparams)
        elif arch == "vit":
            return VITSemi(classes=self.hparams.classes, image_size=self.hparams.test_w, semisupervised=False)
        elif arch == 'deeplabv3':
            return DeepLabv3Semi(hparams=self.hparams)    
        else:
            return None

    # Returns up a new segmentation model and to train the classification layers with a higher learning rate
    def get_new_model(self, save_modules_for_optim=False):
        model, modules_head, modules_back = self.get_new_model_arch()

        if save_modules_for_optim:
            base_lr = self.hparams.optim.lr

            # Project parameters from module groups
            self.params_list = []
            for m in modules_back:
                self.params_list.append({"params": m.parameters(), "lr": base_lr})
            for m in modules_head:
                self.params_list.append({"params": m.parameters(), "lr": base_lr*10})

            self.param_groups_index = len(modules_back)

        return model

    # Setups the segmentation model
    def init_model(self):
        self.model = self.get_new_model(save_modules_for_optim=True)

        if not is_cpu():
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Labelled data
        image, label = batch

        outs = self.model(image)
        pred = outs["pred"]

        # Supervised loss with auxiliary loss
        if self.hparams.aux:
            aux = outs["aux"]
            loss = self.criterion([pred, aux], label)
        else:
            loss = self.criterion(pred, label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        pred = pred.max(1)[1]

        # Optimize the model with the supervised loss
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # Change the learning rate of the model (poly-learning rate)
        self.learning_rate_step(opt, self.hparams.optim.lr, batch_idx)

        # Compute metrics
        intersection, union, target = self.compute_metrics(pred, label)
        self.intersection_meter_train.update(intersection), self.union_meter_train.update(union), self.target_meter_train.update(target)

    # Setups the optimizer for the model
    def configure_optimizers(self):
        if self.hparams.optim.optim == "SGD":
            optimizer = torch.optim.SGD(self.params_list, 
                lr=self.hparams.optim.lr,
                momentum=self.hparams.optim.momentum,
                weight_decay=self.hparams.optim.weight_decay)
        elif self.hparams.optim.optim == "Adam":
            optimizer = torch.optim.Adam(self.params_list, 
                lr=self.hparams.optim.lr,   
                weight_decay=self.hparams.optim.weight_decay)
        else:
            optimizer = None

        assert optimizer is not None, "optimizer type is not supported"

        return optimizer


class SupervisedDataModule(BaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Initialize the data loader for the labelled data
    def train_dataloader(self):
        return DataLoader(self.dataset_train_l, batch_size=self.hparams.batch_size,
                    shuffle=True,
                    num_workers=self.hparams.workers, pin_memory=True,
                    drop_last=True)



def cli_main():
    FlowLightningCLI(SupervisedModel, SupervisedDataModule)

if __name__ == "__main__":
    cli_main()
