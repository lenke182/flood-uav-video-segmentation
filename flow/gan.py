# Implements the s4GAN method for a frame interpolation model
# Sources:
# - https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py
# - https://github.com/hszhao/semseg/blob/master/tool/train.py

import torch
import wandb
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from base.foundation import BetterConcatDataset, is_cpu
from torch.autograd import Variable
from flow.dataset import FlowData
from util.gan import *
from flow.base import FlowBaseModel, FlowBaseDataModule


class FlowGANSemiSupervised(FlowBaseModel):
    
    def __init__(self,
                *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Create and return a new instance of the discriminator
    def get_new_model_arch_D(self):
        return s4GAN_discriminator(num_classes = self.hparams.classes)

    # Initialize the loss functions for the generator and discriminator
    def init_loss_function(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.ignore_index)
        self.criterion_D = nn.BCELoss()

    # Initialize all models, sync batchnorm and ensure classification layers are trained with a higher learning rate
    def init_model(self):
        super().init_model()

        # Init discriminator
        if self.hparams.semisupervised:
            self.model_D = self.get_new_model_arch_D()            
        
        if not is_cpu() and self.hparams.semisupervised:
            self.model_D = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_D)

    # Single training step for the s4GAN method
    # Sources: https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py
    def training_step(self, batch, batch_idx):
        batch_l, batch_u, batch_gt = batch["l"], batch["u"], batch["gt"]
        
        # Labelled data #1
        frame_current_l, frame_prev_l, frame_next_l, mvs_left_l, mvs_right_l, left_index_l, right_index_l, label_l = batch_l["frame_current"], batch_l["frame_prev"], batch_l["frame_next"], batch_l["mvs_left"], batch_l["mvs_right"], batch_l["left_index"], batch_l["right_index"], batch_l["label"]
        
        # Unlabelled data
        frame_current_u, frame_prev_u, frame_next_u, mvs_left_u, mvs_right_u, left_index_u, right_index_u = batch_u["frame_current"], batch_u["frame_prev"], batch_u["frame_next"], batch_u["mvs_left"], batch_u["mvs_right"], batch_u["left_index"], batch_u["right_index"]
        
        # Labelled data #2 (Ground Truth)
        frame_current_gt, label_gt = batch_gt["frame_current"], batch_gt["label"]


        optimizer_G, optimizer_D = self.optimizers()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()


        # Don't train discriminator
        for param in self.model_D.parameters():
            param.requires_grad = False



        # Step 1: supervised loss over labelled data
        pred = self.model_G(frame_current_l, frame_prev_l, frame_next_l, mvs_left_l, mvs_right_l, left_index_l, right_index_l)["pred"]
        loss_ce = self.criterion(pred, label_l)

        intersection, union, target = self.compute_metrics(pred.max(1)[1], label_l)
        self.intersection_meter_train.update(intersection), self.union_meter_train.update(union), self.target_meter_train.update(target)


        # Step 2: self training loss over unlabelled data
        pred_u = self.model_G(frame_current_u, frame_prev_u, frame_next_u, mvs_left_u, mvs_right_u, left_index_u, right_index_u)["pred"]
        frame_current_u = (frame_current_u-torch.min(frame_current_u))/(torch.max(frame_current_u)- torch.min(frame_current_u))
        pred_cat = torch.cat((F.softmax(pred_u, dim=1), frame_current_u), dim=1)
          
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
                
        frame_current_gt = (frame_current_gt - torch.min(frame_current_gt))/(torch.max(frame_current_gt)-torch.min(frame_current_gt))
            
        D_gt_v_cat = torch.cat((D_gt_v, frame_current_gt), dim=1)
        D_out_z_gt , D_out_y_gt = self.model_D(D_gt_v_cat)
        
        # Feature-matching loss between the feature map of the discriminator for the unlabelled data and ground-truth data
        loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))



        # Add all losses for the generator
        if count > 0 and self.get_total_batch_idx(batch_idx) > 0: # if any good predictions found for self-training loss
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
        y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).to(device=frame_prev_l.device))
        loss_d_fake = self.criterion_D(D_out_z, y_fake_) 

        # Real segmentation mask from the ground-truth dataset
        D_out_z_gt , _ = self.model_D(D_gt_v_cat)
        y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).to(device=frame_prev_l.device)) 
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



class FlowGANDataModule(FlowBaseDataModule):

    def __init__(self,
                *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Configure the dataloader for the labelled, unlabelled and ground-truth inputs
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset_train_l = FlowData(split='train', type="l", data_root=self.hparams.data_root,
                                            data_list=self.get_list_path("train.txt"),
                                            transform=self.transform_train, frame_delta=self.hparams.frame_delta,
                                            no_warp=self.hparams.no_warp,no_random_frame_delta=self.hparams.no_random_frame_delta)
            self.dataset_train_u = FlowData(split='train', type="u", data_root=self.hparams.data_root,
                                            data_list=self.get_list_path("train_u.txt"),
                                            transform=self.transform_train, frame_delta=self.hparams.frame_delta,
                                            no_warp=self.hparams.no_warp,no_random_frame_delta=self.hparams.no_random_frame_delta)
            self.dataset_train_gt = FlowData(split='train', type="gt", data_root=self.hparams.data_root,
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
                                                transform=self.transform_predict, frame_delta=self.hparams.frame_delta,
                                                no_warp=self.hparams.no_warp, predict_v_id=self.hparams.predict_v_id)

    # Setup DataLoader classes for the training set
    def train_dataloader(self):

        dataloader_train_l = DataLoader(self.dataset_train_l, batch_size=self.hparams.batch_size,
                   shuffle=True,
                   num_workers=self.hparams.workers, pin_memory=True,
                   prefetch_factor=1, 
                   drop_last=True)

        dataloader_train_u = DataLoader(self.dataset_train_u, batch_size=self.hparams.batch_size,
                    shuffle=True,
                    num_workers=self.hparams.workers, pin_memory=True,
                    prefetch_factor=1,
                    drop_last=True)

        dataloader_train_gt = DataLoader(self.dataset_train_gt, batch_size=self.hparams.batch_size,
                    shuffle=True,
                    num_workers=self.hparams.workers, pin_memory=True,
                    prefetch_factor=1, 
                    drop_last=True)

        return {"l": dataloader_train_l, "u": dataloader_train_u, "gt": dataloader_train_gt}

    # DataLoader for the validation set
    def val_dataloader(self):
        dataloader_val = DataLoader(self.data_val, batch_size=self.hparams.batch_size_val, shuffle=False,
                          num_workers=self.hparams.workers, prefetch_factor=1, pin_memory=True,
        )

        return dataloader_val
