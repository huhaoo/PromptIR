import subprocess
from tqdm import tqdm
import copy
import glob

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
from utils.pytorch_ssim import ssim


def resolve_resume_checkpoint(resume_ckpt, ckpt_dir, auto_resume):
    if resume_ckpt:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")
        return resume_ckpt

    if auto_resume:
        ckpt_candidates = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if not ckpt_candidates:
            return None
        # Pick the most recently modified checkpoint for recovery.
        return max(ckpt_candidates, key=os.path.getmtime)

    return None


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

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        val_loss = self.loss_fn(restored, clean_patch)
        mse = torch.mean((restored - clean_patch) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
        val_psnr = psnr.mean()
        val_ssim = ssim(torch.clamp(restored, 0, 1), torch.clamp(clean_patch, 0, 1), size_average=True)

        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_psnr", val_psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ssim", val_ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if all(k in metrics for k in ("val_loss", "val_psnr", "val_ssim")):
            self.print(
                f"[Val] epoch={self.current_epoch} "
                f"loss={metrics['val_loss'].item():.6f} "
                f"psnr={metrics['val_psnr'].item():.4f} "
                f"ssim={metrics['val_ssim'].item():.4f}"
            )
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=opt.epochs,
        )

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    val_opt = copy.deepcopy(opt)
    val_opt.data_split = "val"
    # Keep full val split without synthetic repeat/downsample.
    val_opt.degradation_size = None
    valset = PromptTrainDataset(val_opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    valloader = DataLoader(valset, batch_size=opt.batch_size, pin_memory=True, shuffle=False,
                           drop_last=False, num_workers=opt.num_workers)
    
    model = PromptIRModel()

    if torch.cuda.is_available() and opt.num_gpus > 0:
        devices = min(opt.num_gpus, torch.cuda.device_count())
        trainer_kwargs = dict(max_epochs=opt.epochs, accelerator="gpu", devices=devices,
                              logger=logger, callbacks=[checkpoint_callback],
                              accumulate_grad_batches=opt.accumulate_grad_batches,
                              check_val_every_n_epoch=4,
                              num_sanity_val_steps=0)
        if devices > 1:
            trainer_kwargs["strategy"] = "ddp"
    else:
        trainer_kwargs = dict(max_epochs=opt.epochs, accelerator="cpu", devices=1,
                              logger=logger, callbacks=[checkpoint_callback],
                              accumulate_grad_batches=opt.accumulate_grad_batches,
                              check_val_every_n_epoch=4,
                              num_sanity_val_steps=0)

    trainer = pl.Trainer(**trainer_kwargs)
    resume_path = resolve_resume_checkpoint(opt.resume_ckpt, opt.ckpt_dir, opt.auto_resume)
    if resume_path is not None:
        print(f"Resuming training from checkpoint: {resume_path}")

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=resume_path,
    )


if __name__ == '__main__':
    main()



