# Implements the s4GAN method for the segmentation task without frame interpolation
# Sources:
# - https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py
# - https://github.com/hszhao/semseg/blob/master/tool/train.py



from base.cli import FlowLightningCLI

import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model.pspnet import PSPNet
from model.deeplabv3 import DeepLabv3
from model.vit import VITSegmentModel

from base.foundation import BaseModel, BaseDataModule, is_cpu, round_train

from util.gan import *

# Hyperparameter related to the optimizer 
@dataclass_json
@dataclass
class OptimKWArgs(dict):
    optim: str = "SGD"
    lr: float = 2.5e-4
    lr_D: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.0005

# Hyperparameter related to the loss function
@dataclass_json
@dataclass
class LossKWArgs(dict):
    loss: str = "cross_entropy"

class GANSemiSupervised(BaseModel):

    def __init__(self, 
                test_h: int = 873,
                test_w: int = 873,
                arch: str = "pspnet",
                aux: bool = True,
                loss: LossKWArgs = LossKWArgs(),
                optim: OptimKWArgs = OptimKWArgs(),
                semisupervised: bool = False,
                threshold_st = 0.6,
                lambda_fm = 0.1,
                lambda_st = 1.0,
                pretrained: bool = True,
                layers: int = 101,
                zoom_factor: int = 8,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        loss = loss.to_dict()
        optim = optim.to_dict()
        test_w = round_train(test_w, arch)
        test_h = round_train(test_h, arch)
        self.save_hyperparameters()
        self.hparams.test_w = round_train(test_w, arch)
        self.hparams.test_h = round_train(test_h, arch)
        self.hparams.loss = LossKWArgs.from_dict(loss)
        self.hparams.optim = OptimKWArgs.from_dict(optim)

    # Initialize the loss functions for the generator and discriminator
    def init_loss_function(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.ignore_index)
        self.criterion_D = nn.BCELoss()
        
    # Create and return a new instance of the generator
    def get_new_model_arch_G(self):
        # Init G
        arch = self.hparams.arch

        self.hparams.test_w = round_train(self.hparams.test_w, self.hparams.arch)
        self.hparams.test_h = round_train(self.hparams.test_h, self.hparams.arch)

        if arch == "pspnet":
            model = PSPNet(hparams=self.hparams)
            modules_head = [model.ppm, model.cls, model.aux]
            modules_back = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        elif arch == "vit":
            model = VITSegmentModel(self.hparams.classes, self.hparams.test_w)
            modules_head = [model.model.decoder]
            modules_back = [model.model.encoder]
        elif arch == 'deeplabv3':
            model = DeepLabv3(hparams=self.hparams)
            modules_head = [model.model.classifier, model.model.aux_classifier]
            modules_back = [model.model.backbone]
        else:
            return None

        return model, modules_head, modules_back

    # Create and return a new instance of the discriminator
    def get_new_model_arch_D(self):
        model_D = s4GAN_discriminator(num_classes = self.hparams.classes)

        return model_D

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

    def forward(self, x):        
        return self.model_G(x)

    # Single training step for the s4GAN method
    # Sources: https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py
    def training_step(self, batch, batch_idx):
        batch_l, batch_u, batch_gt = batch["l"], batch["u"], batch["gt"]

        # Labelled data #1
        image_l, label_l = batch_l
        # Unlabelled data
        image_u, _ = batch_u
        # Labelled data #2 (Ground Truth)
        image_gt, label_gt = batch_gt


        optimizer_G, optimizer_D = self.optimizers()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # Don't train discriminator
        for param in self.model_D.parameters():
            param.requires_grad = False


        # Step 1: supervised loss over labelled data
        pred = self.model_G(image_l)["pred"]
        loss_ce = self.criterion(pred, label_l)

        intersection, union, target = self.compute_metrics(pred.max(1)[1], label_l)
        self.intersection_meter_train.update(intersection), self.union_meter_train.update(union), self.target_meter_train.update(target)


        # Step 2: self training loss over unlabelled data
        pred_u = self.model_G(image_u)["pred"]

        image_u = (image_u-torch.min(image_u))/(torch.max(image_u)- torch.min(image_u))
        pred_cat = torch.cat((F.softmax(pred_u, dim=1), image_u), dim=1)
          
        # Returns the confidence and the feature map of the discriminator 
        D_out_z, D_out_y_pred = self.model_D(pred_cat)
  
        # Select predictions above confidence threshold
        pred_sel, labels_sel, count = find_good_maps(D_out_z, pred_u, self.hparams.threshold_st) 
        self.log("train_st_threshold", count, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Self training loss
        if count > 0 and self.get_total_batch_idx(batch_idx) > 0:
            loss_st = self.criterion(pred_sel, labels_sel)
        else:
            loss_st = 0.0


        # STEP 3: Discriminator receives ground-truth segmentation maps as input
        D_gt_v = one_hot(label_gt, self.hparams.classes)
                
        image_gt = (image_gt - torch.min(image_gt))/(torch.max(image_l)-torch.min(image_l))
            
        D_gt_v_cat = torch.cat((D_gt_v, image_gt), dim=1)
        D_out_z_gt, D_out_y_gt = self.model_D(D_gt_v_cat)
        
        # Feature-matching loss between the feature map of the discriminator for the unlabelled data and ground-truth data
        loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))



        # Add all losses for the generator
        if count > 0 and self.get_total_batch_idx(batch_idx) > 0:
            loss_s = loss_ce + self.hparams.lambda_fm*loss_fm + self.hparams.lambda_st*loss_st 
        else:
            loss_s = loss_ce + self.hparams.lambda_fm*loss_fm

        self.manual_backward(loss_s)


        # STEP 4: Train the discriminator
        for param in self.model_D.parameters():
            param.requires_grad = True

        # Fake input data generated by the generator
        pred_cat = pred_cat.detach()
        
        D_out_z, _ = self.model_D(pred_cat)
        y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).to(device=image_l.device))
        loss_d_fake = self.criterion_D(D_out_z, y_fake_) 

        # Real segmentation mask from the ground-truth dataset
        D_out_z_gt , _ = self.model_D(D_gt_v_cat)
        y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).to(device=image_l.device)) 
        loss_d_real = self.criterion_D(D_out_z_gt, y_real_)
        
        # Discriminator loss
        loss_d = (loss_d_fake + loss_d_real)/2.0
        self.manual_backward(loss_d)

        # Log values locally and to Weights and Biases
        self.log("train_loss", loss_s+loss_d, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_s", loss_s, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_ce", loss_ce, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_fm", loss_fm, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_st", loss_st, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_d", loss_d, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Optimize the generator and discriminator based on the loss functions
        optimizer_G.step()
        optimizer_D.step()

        # Change the learning rate for the generator and discriminator (poly-learning rate)
        self.learning_rate_step(optimizer_G, self.hparams.optim.lr, batch_idx)
        self.learning_rate_step(optimizer_D, self.hparams.optim.lr_D, batch_idx)

    # Define the optimizers for the generator and discriminator
    def configure_optimizers(self):
        optimizer_G = torch.optim.SGD(self.model_G_params_list, 
            lr=self.hparams.optim.lr,
            momentum=self.hparams.optim.momentum,
            weight_decay=self.hparams.optim.weight_decay)
        
        optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=self.hparams.optim.lr_D, betas=(0.9,0.99))
    
        return optimizer_G, optimizer_D



class GANDataModule(BaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Configure the dataloader for the labelled, unlabelled and ground-truth inputs
    def train_dataloader(self):
        if self.hparams.semisupervised:
            dataloader_train_l = DataLoader(self.dataset_train_l, batch_size=self.hparams.batch_size,
                        shuffle=True,
                        num_workers=self.hparams.workers, pin_memory=True,
                        drop_last=True)
            dataloader_train_u = DataLoader(self.dataset_train_u, batch_size=self.hparams.batch_size,
                        shuffle=True,
                        num_workers=self.hparams.workers, pin_memory=True,
                        drop_last=True)
            dataloader_train_gt = DataLoader(self.dataset_train_l, batch_size=self.hparams.batch_size,
                        shuffle=True,
                        num_workers=self.hparams.workers, pin_memory=True,
                        drop_last=True)

            return {"l": dataloader_train_l, "u": dataloader_train_u, "gt": dataloader_train_gt}
        else:
            return DataLoader(self.dataset_train_l, batch_size=self.hparams.batch_size,
                        shuffle=True,
                        num_workers=self.hparams.workers, pin_memory=True,
                        drop_last=True)

def cli_main():
    FlowLightningCLI(GANSemiSupervised, GANDataModule)

if __name__ == "__main__":
    cli_main()
