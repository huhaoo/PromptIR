import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()

    if torch.cuda.is_available() and opt.num_gpus > 0:
        devices = min(opt.num_gpus, torch.cuda.device_count())
        trainer_kwargs = dict(max_epochs=opt.epochs, accelerator="gpu", devices=devices,
                              logger=logger, callbacks=[checkpoint_callback])
        if devices > 1:
            trainer_kwargs["strategy"] = "ddp"
    else:
        trainer_kwargs = dict(max_epochs=opt.epochs, accelerator="cpu", devices=1,
                              logger=logger, callbacks=[checkpoint_callback])

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



