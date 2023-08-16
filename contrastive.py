# Sources:
# - https://github.com/Haochen-Wang409/U2PL/blob/main/train_semi.py
# - https://github.com/hszhao/semseg/blob/master/tool/train.py

# Implements the U2PL method for the segmentation task without frame interpolation

from base.cli import FlowLightningCLI

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from model.pspnet import PSPNetSemi
from model.vit import VITSemi
from model.deeplabv3 import DeepLabv3Semi

from u2pl.utils.loss_helper import CriterionOhem
from u2pl.dataset.augmentation import generate_unsup_data
from u2pl.utils.loss_helper import compute_contra_memobank_loss, compute_unsupervised_loss
from u2pl.utils.utils import label_onehot

from base.foundation import BaseModel, BaseDataModule, is_cpu, round_train

# Hyperparameter related to the contrastive loss
@dataclass_json
@dataclass
class ContrastiveKWArgs(dict):
    enabled: bool = True
    negative_high_entropy: bool = True
    low_rank: int = 3
    high_rank: int = 20
    current_class_threshold: float = 0.3
    current_class_negative_threshold: int = 1
    unsupervised_entropy_ignore: int = 80
    low_entropy_threshold: int = 20
    num_negatives: int = 50
    num_queries: int = 256
    temperature: float = 0.5
    loss_weight: int = 1

# Hyperparameter related to the optimizer 
@dataclass_json
@dataclass
class OptimKWArgs(dict):
    optim: str = "SGD"
    lr: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 0.0001

# Hyperparameter related to the loss function
@dataclass_json
@dataclass
class LossKWArgs(dict):
    loss: str = "ohem"
    thresh: float = 0.7
    min_kept: int = 100000


class ContrastiveModel(BaseModel):

    def __init__(self,
                 arch: str = "pspnet",
                 test_h: int = 873,
                 test_w: int = 873,
                 layers: int = 101,
                 zoom_factor: int = 8,
                 pretrained: bool = True,
                 aux: bool = True,
                 aux_weight=0.4,
                 loss: LossKWArgs = LossKWArgs(),
                 optim: OptimKWArgs = OptimKWArgs(),
                 sup_only_epoch = 0,
                 contrastive: ContrastiveKWArgs = ContrastiveKWArgs(),
                 unsupervised_apply_aug = "cutmix",
                 unsupervised_drop_percent = 80, #80
                 unsupervised_loss_weight = 1.0,
                 semisupervised = True,
                 ema_decay = 0.99,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure parameters are correctly logged by Weights and Biases
        loss = loss.to_dict()
        optim = optim.to_dict()
        contrastive = contrastive.to_dict()
        test_w = round_train(test_w, arch)
        test_h = round_train(test_h, arch)
        self.save_hyperparameters()
        self.hparams.test_w = round_train(test_w, arch)
        self.hparams.test_h = round_train(test_h, arch)
        self.hparams.loss = LossKWArgs.from_dict(loss)
        self.hparams.optim = OptimKWArgs.from_dict(optim)
        self.hparams.contrastive = ContrastiveKWArgs.from_dict(contrastive)

        
        self.init_unsupervised_params()

    # Initialize the loss functions
    def init_loss_function(self):
        aux_weight = self.hparams.aux_weight if self.hparams.aux else 0

        self.criterion = CriterionOhem(
            aux_weight,
            ignore_index=self.hparams.ignore_index,
            thresh = self.hparams.loss.thresh,
            min_kept = self.hparams.loss.min_kept
        )

    # Returns a new model instance
    def get_new_model_arch(self):
        arch = self.hparams.arch
        
        self.hparams.test_w = round_train(self.hparams.test_w, self.hparams.arch)
        self.hparams.test_h = round_train(self.hparams.test_h, self.hparams.arch)

        if arch == "pspnet":
            return PSPNetSemi(hparams=self.hparams)
        elif arch == "vit":
            return VITSemi(classes=self.hparams.classes, image_size=self.hparams.test_w, semisupervised=True)
        elif arch == 'deeplabv3':
            return DeepLabv3Semi(hparams=self.hparams)     
        else:
            return None

    # Sets up a new model instance and ensures that the classifications layers are trained with a higher learning rate
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

    # Initialize the student and teacher network
    def init_model(self):
        self.model = self.get_new_model(save_modules_for_optim=True)

        if self.hparams.semisupervised:
            self.model_teacher = self.get_new_model(save_modules_for_optim=False)

            for p in self.model_teacher.parameters():
                p.requires_grad = False
        
        if not is_cpu():
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.hparams.semisupervised:
                self.model_teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_teacher)

    # Initialize the memory bank
    # Source: https://github.com/Haochen-Wang409/U2PL/blob/main/train_semi.py
    def init_unsupervised_params(self):
        if self.hparams.semisupervised:
            # build class-wise memory bank
            self.memobank = []
            self.queue_ptrlis = []
            self.queue_size = []
            for i in range(self.hparams.classes):
                self.memobank.append([torch.zeros(0, 256)])
                self.queue_size.append(30000)
                self.queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
            self.queue_size[0] = 50000

    def forward(self, x):
        if self.trainer.current_epoch < self.hparams.sup_only_epoch:
            return self.model(x)
        else:
            return self.model_teacher(x)

    # Source: https://github.com/Haochen-Wang409/U2PL/blob/main/train_semi.py
    def training_step(self, batch, batch_idx):
        batch_l, batch_u = batch["l"], batch["u"]

        # Labelled data
        image_l, label_l = batch_l
        # Unlabelled data
        image_u, _ = batch_u

        # Initially, train the student network supervised for 2 epochs
        if self.trainer.current_epoch < self.hparams.sup_only_epoch:
            outs = self.model(image_l)
            pred, rep = outs["pred"], outs["rep"]

            # Supervised loss
            if self.hparams.aux:
                aux = outs["aux"]
                sup_loss = self.criterion([pred, aux], label_l)
            else:
                sup_loss = self.criterion(pred, label_l)

            # Compute metrics
            intersection, union, target = self.compute_metrics(pred.max(1)[1], label_l)
            self.intersection_meter_train.update(intersection), self.union_meter_train.update(union), self.target_meter_train.update(target)

            self.model_teacher.train()
            _ = self.model_teacher(image_l)
            
            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            # After the second epoch, copy the parameters of the student network to the teacher network
            if self.trainer.current_epoch == self.hparams.sup_only_epoch:
                with torch.no_grad():
                    for t_params, s_params in zip(
                        self.model_teacher.parameters(), self.model.parameters()
                    ):
                        t_params.data = s_params.data

            # Compute the pseudo-labels with the teacher network
            self.model_teacher.eval()
            pred_u_teacher = self.model_teacher(image_u)["pred"]
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # Heavy data augmentation to the unsupervised images and labels
            if np.random.uniform(0, 1) < 0.5 and self.hparams.unsupervised_apply_aug:
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=self.hparams.unsupervised_apply_aug,
                )
            else:
                image_u_aug = image_u

            # Predict output of the student network for the labelled and unlabelled data
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))      

            outs = self.model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]

            pred_l_large, pred_u_large = pred_all[:num_labeled], pred_all[num_labeled:]    

            # Compute the supervised loss
            if self.hparams.aux:
                aux = outs["aux"][:num_labeled]
                sup_loss = self.criterion([pred_l_large, aux], label_l.clone())
            else:
                sup_loss = self.criterion(pred_l_large, label_l.clone())

            # Compute the metrics
            intersection, union, target = self.compute_metrics(pred_l_large.max(1)[1], label_l)
            self.intersection_meter_train.update(intersection), self.union_meter_train.update(union), self.target_meter_train.update(target)

            # Predict the output of the teacher network for the labelled and unlabelled data
            self.model_teacher.train()
            with torch.no_grad():
                out_t = self.model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )

                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = pred_u_teacher

            # Compute the unsupervised loss
            drop_percent = self.hparams.unsupervised_drop_percent
            percent_unreliable = (100 - drop_percent) * (1 - float(self.trainer.current_epoch) / self.trainer.max_epochs)
            drop_percent = 100 - percent_unreliable
            unsup_loss = (
                    compute_unsupervised_loss(
                        pred_u_large,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher.detach(),
                    )
                    * self.hparams.unsupervised_loss_weight
            )

            # Compute the contrastive loss
            contra_flag = "none"
            if self.hparams.contrastive.enabled:
                cfg_contra = self.hparams.contrastive
                contra_flag = "{}:{}".format(
                    cfg_contra.low_rank, cfg_contra.high_rank
                )
                alpha_t = cfg_contra.low_entropy_threshold * (
                    1 - self.trainer.current_epoch / self.trainer.max_epochs
                )

                # Select the appropriate pixels from the prediction of the teacher network, the labels and the pseudo-labels
                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                        entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                    )

                    low_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )

                    low_mask_all = F.interpolate(
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )

                    if cfg_contra.negative_high_entropy:
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u_aug.shape)
                                .float()
                                .unsqueeze(1)
                                .to(device=image_l.device),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  

                    label_l_small = F.interpolate(
                        label_onehot(label_l, self.hparams.classes),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    ).long()
                    label_u_small = F.interpolate(
                        label_onehot(label_u_aug, self.hparams.classes),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    ).long()

                # Compute the contrastive loss with the memory bank
                new_keys, contra_loss = compute_contra_memobank_loss(
                    rep_all,
                    label_l_small,
                    label_u_small,
                    prob_l_teacher.detach(),
                    prob_u_teacher.detach(),
                    low_mask_all,
                    high_mask_all,
                    cfg_contra,
                    self.memobank,
                    self.queue_ptrlis,
                    self.queue_size,
                    rep_all_teacher.detach(),
                )

                contra_loss = (
                    contra_loss 
                    / torch.cuda.device_count()
                    * cfg_contra.loss_weight
                )

            else:
                contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_unsup_loss", unsup_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_contra_loss", contra_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Optimize the student network
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.learning_rate_step(opt, self.hparams.optim.lr, batch_idx)
        
        # Change the parameters of the teacher network to reflect the exponential moving average of the student network
        if self.trainer.current_epoch >= self.hparams.sup_only_epoch:
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        self.get_total_batch_idx(batch_idx)
                        - self.trainer.estimated_stepping_batches * self.hparams.sup_only_epoch
                        + 1
                    ),
                    self.hparams.ema_decay,
                )
                for t_params, s_params in zip(
                    self.model_teacher.parameters(), self.model.parameters()
                ):
                    t_params.data.copy_(
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )
    
    # Setup the optimizers for the student network
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


class ContrastiveDataModule(BaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Setup the dataloader for the labelled and unlabelled data
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

            return {"l": dataloader_train_l, "u": dataloader_train_u}
        else:
            return DataLoader(self.dataset_train_l, batch_size=self.hparams.batch_size,
                        shuffle=True,
                        num_workers=self.hparams.workers, pin_memory=True,
                        drop_last=True)

def cli_main():
    FlowLightningCLI(ContrastiveModel, ContrastiveDataModule)

if __name__ == "__main__":
    cli_main()