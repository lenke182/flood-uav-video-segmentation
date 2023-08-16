# Sources:
# - https://github.com/tshu-w/lightning-template/commit/f1386e05e6f01d71d4c1717fe10673cfd243acfe#diff-8af0b134246b678d4a20a3bafb46020192aa34a1aa437dc9b193837236254007

import uuid
import numpy as np
import os
import pytorch_lightning as pl
import torch
import wandb
import sys
import json
from collections import ChainMap
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.metrics import metrics_to_scalars
from util.util import colorize

from base.foundation import is_cpu

# Callback that logs a number of test images, labels and predictions to Weights and Biases
# Predictions and labels are colored before being uploaded
class LogWandBPredictionsCallback(Callback):

    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        # Store data between batches in memory
        self.data = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if outputs is not None:
            batch_test, test_idx = batch
            if isinstance(batch_test, dict):
                x = batch_test["frame_current"]
                y = batch_test["label"]
            else:
                x, y = batch_test

            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            assert x.shape[0] == 1 and y.shape[0] == 1

            data_root = pl_module.hparams.data_root
            colors_path = os.path.join(data_root, "list", "colors.txt")

            colors = np.loadtxt(colors_path).astype('uint8')

            y = np.squeeze(np.uint8(y), axis=0)
            outputs = np.uint8(outputs.detach().cpu())
            y_color = colorize(y, colors)
            outputs_color = colorize(outputs, colors)

            image = np.squeeze(x, axis=0)

            image_wandb = wandb.Image(image.transpose(1, 2, 0))
            y_color_wandb = wandb.Image(y_color)
            outputs_color_wandb = wandb.Image(outputs_color)

            self.data.append([image_wandb, y_color_wandb, outputs_color_wandb])

    # Upload table
    def on_test_end(self, *args, **kwargs):
        self.wandb_logger.log_table(key='test_outputs', columns=['image', 'ground truth', 'prediction'], data=self.data)


# Extends the PyTorch Lightning CLI with additional functionality
class FlowLightningCLI(LightningCLI):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        # Log command line arguments
        arguments = sys.argv[1:]
        sys.argv = ['']
        print(arguments)
        super().__init__(args=arguments, *args, **kwargs)

    # Extend the PyTorch Lightning CLI with additional arguments
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--wandb", type=str, default=None)
        parser.add_argument("--runid", type=str, default=None)
        parser.add_argument("--tag", default=[], action='append')
        parser.link_arguments("data.data_root", "model.data_root")
        parser.link_arguments("data.train_w", "data.train_h")
        parser.link_arguments("data.train_w", "model.test_w")
        parser.link_arguments("data.train_w", "model.test_h")
        parser.link_arguments("model.no_cropping", "data.no_cropping")
        parser.link_arguments("model.no_warp", "data.no_warp")
        parser.link_arguments("model.arch", "data.arch")
        parser.link_arguments("data.semisupervised", "model.semisupervised")
        parser.link_arguments("data.predict_v_id", "model.predict_v_id")

    # Initialize the trainer by setting callbacks and loggers
    def instantiate_trainer(self, **kwargs):
        if "callbacks" not in kwargs:
            kwargs["callbacks"] = []

        if "logger" not in kwargs:
            kwargs["logger"] = []

        kwargs["callbacks"].append(EarlyStopping(monitor="val_miou_epoch", mode="max", min_delta=0.001, patience=10, strict=False))


        # If Weights and Biases is enabled        
        wandb_project = self.config[self.config.subcommand]["wandb"]
        if wandb_project:
            wandb_kwargs = {
                "project": wandb_project
            }

            # Resume run if runid present
            runid = self.config[self.config.subcommand]["runid"]
            if runid:
                wandb_kwargs["id"] = runid

            # Add tags to run
            tag = self.config[self.config.subcommand]["tag"]
            if tag:
                wandb_kwargs["tags"] = tag
            
            wandb_logger = pl.loggers.WandbLogger(**wandb_kwargs)
            name = str(wandb_logger.experiment.id)

            # Enable Weights and Biases prediction callback
            #kwargs["callbacks"].append(LogWandBPredictionsCallback(wandb_logger))
        else:
            name = str(uuid.uuid4())

        # Add loggers
        kwargs["logger"].append(TensorBoardLogger("logs", name=name))
        if wandb_project:
            kwargs["logger"].append(wandb_logger)

        # Save model weights each epoch
        kwargs["callbacks"].append(ModelCheckpoint(
            monitor='val_miou_epoch',
            mode="max",
            dirpath=os.path.join("logs", name),
            save_top_k=5,
            filename='{epoch}-{val_miou_epoch:.2f}',
            save_on_train_epoch_end=False,
            save_last=True,
            every_n_train_steps=None,
            every_n_epochs=1
        ))

        # Enables the PyTorch Profiler
        #kwargs["profiler"] = pl.profilers.PyTorchProfiler(row_limit=-1, filename="profile.txt", sort_by_key="cuda_time_total")

        # Set CPU or GPU relevant parameters
        if is_cpu():
            kwargs["devices"] = 1
            kwargs["strategy"] = None
            kwargs["sync_batchnorm"] = False
        else:
            kwargs["devices"] = torch.cuda.device_count()

        return super(FlowLightningCLI, self).instantiate_trainer(**kwargs)

    # After running train and validation (fit), run testing (test) and inference (predict)
    # Saves test metrics locally
    # Source: https://github.com/tshu-w/lightning-template/commit/f1386e05e6f01d71d4c1717fe10673cfd243acfe#diff-8af0b134246b678d4a20a3bafb46020192aa34a1aa437dc9b193837236254007 (Code was lightly modified)
    def after_run(self) -> None:
        results = {}

        if self.trainer.state.fn == pl.trainer.states.TrainerFn.FITTING and self.trainer.checkpoint_callback and self.trainer.checkpoint_callback.best_model_path:
            fn_kwargs = {
                "model": self.model,
                "datamodule": self.datamodule,
                "ckpt_path": self.trainer.checkpoint_callback.best_model_path,
            }
            has_test_loader = self.trainer._data_connector._test_dataloader_source.is_defined()

            print("has_test_loader:", has_test_loader)

            test_results = self.trainer.test(**fn_kwargs) if has_test_loader else []

            results = dict(ChainMap(*test_results))
                            
            has_predict_loader = self.trainer._data_connector._predict_dataloader_source.is_defined()

            print("has_predict_loader:", has_predict_loader)

            self.trainer.predict(**fn_kwargs) if has_predict_loader else []
        else:
            results = metrics_to_scalars(self.trainer.logged_metrics)

        if results:
            results_str = json.dumps(results, ensure_ascii=False, indent=2)

            metrics_file = os.path.join(self.trainer.log_dir, "metrics.json")
            with open(metrics_file, "w") as f:
                f.write(results_str)

    after_fit = after_validate = after_test = after_run