import copy
import glob
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import StrategyRegistry
from torch.utils.data import DataLoader

from net.model import build_promptir_model_from_options
from options import options as opt
from utils.dataset_utils import PromptTrainDataset
from utils.pytorch_ssim import ssim
from utils.schedulers import LinearWarmupCosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from promptir_adv_training import promptir_adv_mix_dataset


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


class checkpoint_epoch_step_plus_one(ModelCheckpoint):
    def format_checkpoint_name(self, metrics, filename=None, ver=None) -> str:
        adjusted = dict(metrics)

        if "epoch" in adjusted:
            try:
                adjusted["epoch"] = int(adjusted["epoch"]) + 1
            except Exception:
                pass
        if "step" in adjusted:
            try:
                adjusted["step"] = int(adjusted["step"]) + 1
            except Exception:
                pass

        return super().format_checkpoint_name(adjusted, filename=filename, ver=ver)


class PromptIRAdvModel(pl.LightningModule):
    def __init__(self, train_dataset=None):
        super().__init__()
        self.net = build_promptir_model_from_options(opt, decoder=True)
        self.loss_fn = nn.L1Loss()
        self.train_dataset = train_dataset

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
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

    def on_train_epoch_start(self):
        if self.train_dataset is None:
            return

        model_was_training = self.net.training
        self.net.eval()

        force = self.current_epoch == 0
        try:
            if hasattr(self.train_dataset, "resample_main_process_then_sync"):
                self.train_dataset.resample_main_process_then_sync(
                    epoch=int(self.current_epoch),
                    force=force,
                    promptir_model_override=self.net,
                )
                return
            if hasattr(self.train_dataset, "resample_adversarial"):
                self.train_dataset.resample_adversarial(
                    epoch=int(self.current_epoch),
                    force=force,
                    promptir_model_override=self.net,
                )
        finally:
            if model_was_training:
                self.net.train()

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

    if not bool(opt.adv_enable):
        print("[adv_train] forcing adv_enable=True for this entrypoint")
        opt.adv_enable = True

    # Keep adversarial resample cache scoped to the checkpoint directory.
    adv_cache_root = Path(opt.ckpt_dir).expanduser().resolve() / "adv_data"
    opt.adv_cache_root = str(adv_cache_root)
    print(f"[adv_train] adv_cache_root is pinned to: {opt.adv_cache_root}")

    if opt.wblogger is not None:
        run_name = str(getattr(opt, "wandb_run_name", "")).strip() or "PromptIR-AdvTrain"
        logger = WandbLogger(project=opt.wblogger, name=run_name)
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset_base = PromptTrainDataset(opt)
    trainset = promptir_adv_mix_dataset(trainset_base, opt)

    val_opt = copy.deepcopy(opt)
    val_opt.data_split = "val"
    val_opt.degradation_size = None
    valset = PromptTrainDataset(val_opt)

    checkpoint_callback = checkpoint_epoch_step_plus_one(
        dirpath=opt.ckpt_dir,
        filename="epoch={epoch}-step={step}",
        auto_insert_metric_name=False,
        every_n_epochs=8,
        save_top_k=-1,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )
    valloader = DataLoader(
        valset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers,
    )

    model = PromptIRAdvModel(train_dataset=trainset)

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
