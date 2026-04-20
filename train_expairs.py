import copy
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import StrategyRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTIR_ROOT = PROJECT_ROOT / "PromptIR"
if str(PROMPTIR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPTIR_ROOT))

from net.model import build_promptir_model_from_options
from options import options as opt
from utils.dataset_utils import PromptTrainDataset
from utils.pytorch_ssim import ssim
from utils.schedulers import LinearWarmupCosineAnnealingLR


def resolve_resume_checkpoint(resume_ckpt, ckpt_dir, auto_resume):
    if resume_ckpt:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")
        return resume_ckpt

    if auto_resume:
        ckpt_candidates = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if not ckpt_candidates:
            return None
        return max(ckpt_candidates, key=os.path.getmtime)

    return None


def select_multi_gpu_strategy():
    available = set(StrategyRegistry.available_strategies())
    candidates = [
        "ddp_find_unused_parameters_true",
        "ddp",
        "ddp_find_unused_parameters_false",
        "ddp_spawn",
    ]
    for name in candidates:
        if name in available:
            return name
    return None


class PromptIRExtraPairModel(pl.LightningModule):
    def __init__(self, p_target_m_target=0.1, p_target_mm_input=0.1):
        super().__init__()
        self.net = build_promptir_model_from_options(opt, decoder=True)
        self.loss_fn = nn.L1Loss()
        self.p_target_m_target = float(p_target_m_target)
        self.p_target_mm_input = float(p_target_mm_input)
        self.p_input = 1.0 - self.p_target_m_target - self.p_target_mm_input
        if self.p_input < 0.0:
            raise ValueError("Invalid probabilities: require p_target_m_target + p_target_mm_input <= 1.0")

    def forward(self, x):
        return self.net(x)

    def _batch_l1_per_sample(self, pred, target):
        return F.l1_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        batch_size = degrad_patch.shape[0]

        # Per-sample replacement policy:
        # - p_input: M(input)-target
        # - p_target_m_target: M(target)-target
        # - p_target_mm_input: M(M(input))-target
        rand_vals = torch.rand(batch_size, device=degrad_patch.device)
        threshold_input = self.p_input
        threshold_target = self.p_input + self.p_target_m_target

        mask_input = rand_vals < threshold_input
        mask_target = (rand_vals >= threshold_input) & (rand_vals < threshold_target)
        mask_mm_input = rand_vals >= threshold_target

        selected_loss_per_sample = torch.empty(batch_size, device=degrad_patch.device)

        replace_input_count = int(mask_input.sum().item())
        if replace_input_count > 0:
            m_input_batch = self.net(degrad_patch[mask_input])
            input_loss_per_sample = self._batch_l1_per_sample(m_input_batch, clean_patch[mask_input])
            selected_loss_per_sample[mask_input] = input_loss_per_sample

        replace_m_target_count = int(mask_target.sum().item())
        if replace_m_target_count > 0:
            m_target_batch = self.net(clean_patch[mask_target])
            target_loss_per_sample = self._batch_l1_per_sample(m_target_batch, clean_patch[mask_target])
            selected_loss_per_sample[mask_target] = target_loss_per_sample

        replace_mm_input_count = int(mask_mm_input.sum().item())
        if replace_mm_input_count > 0:
            m_input_batch_for_mm = self.net(degrad_patch[mask_mm_input])
            mm_input_batch = self.net(m_input_batch_for_mm)
            mm_input_loss_per_sample = self._batch_l1_per_sample(mm_input_batch, clean_patch[mask_mm_input])
            selected_loss_per_sample[mask_mm_input] = mm_input_loss_per_sample

        loss = selected_loss_per_sample.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_effective_samples", float(batch_size), on_step=True, on_epoch=True)
        self.log("train_replace_input_count", float(replace_input_count), on_step=True, on_epoch=True)
        self.log("train_replace_m_target_count", float(replace_m_target_count), on_step=True, on_epoch=True)
        self.log("train_replace_mm_input_count", float(replace_mm_input_count), on_step=True, on_epoch=True)
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

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=opt.epochs,
        )
        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)

    if opt.wblogger is not None:
        run_name = str(getattr(opt, "wandb_run_name", "")).strip() or "PromptIR-Train-ExtraPairs"
        logger = WandbLogger(project=opt.wblogger, name=run_name)
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)

    val_opt = copy.deepcopy(opt)
    val_opt.data_split = "val"
    val_opt.degradation_size = None
    valset = PromptTrainDataset(val_opt)

    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers,
    )

    model = PromptIRExtraPairModel(
        p_target_m_target=opt.p_target_m_target,
        p_target_mm_input=opt.p_target_mm_input,
    )

    if torch.cuda.is_available() and opt.num_gpus > 0:
        devices = min(opt.num_gpus, torch.cuda.device_count())
        trainer_kwargs = dict(
            max_epochs=opt.epochs,
            accelerator="gpu",
            devices=devices,
            logger=logger,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=opt.accumulate_grad_batches,
            check_val_every_n_epoch=4,
            num_sanity_val_steps=0,
        )
        if devices > 1:
            strategy_name = select_multi_gpu_strategy()
            if strategy_name is not None:
                print(f"Using multi-GPU strategy: {strategy_name}")
                trainer_kwargs["strategy"] = strategy_name
    else:
        trainer_kwargs = dict(
            max_epochs=opt.epochs,
            accelerator="cpu",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=opt.accumulate_grad_batches,
            check_val_every_n_epoch=4,
            num_sanity_val_steps=0,
        )

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


if __name__ == "__main__":
    main()
