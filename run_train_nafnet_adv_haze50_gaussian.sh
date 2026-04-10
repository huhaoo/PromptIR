#!/usr/bin/env bash
set -euo pipefail

# NAFNet adversarial dehaze training launcher with gaussian control-map interpolation.
# This launcher keeps the existing PromptIR/adv_train.py entrypoint and injects
# interpolation settings through explicit CLI arguments.

cd /home/huhao/adv_ir/PromptIR

echo "[Gaussian NAFNet Adv Train] mode=gaussian radius=4 sigma=1.25 extra=2"

# WANDB_MODE="offline" \
conda run -n promptir --no-capture-output \
  python adv_train.py \
  --model_arch "nafnet" \
  --de_type dehaze \
  --epochs "128" \
  --batch_size "4" \
  --accumulate_grad_batches "2" \
  --lr "2e-4" \
  --patch_size "128" \
  --num_workers "8" \
  --num_gpus "3" \
  --ckpt_dir "/home/huhao/adv_ir/exp/train_ckpt_nafnet_adv_haze50_gaussian" \
  --adv_ratio "0.5" \
  --adv_resample_epochs "4" \
  --adv_steps1 "4" \
  --adv_steps2 "16" \
  --adv_step_size "0.03" \
  --adv_lambda_reg "0.05" \
  --adv_promptir_patch_size "256" \
  --adv_promptir_patch_overlap "32" \
  --adv_cache_root "/home/huhao/adv_ir/PromptIR/data/Train/adv_pairs" \
  --adv_attack_device "cuda" \
  --adv_map_interp_mode "gaussian" \
  --adv_gaussian_radius "4" \
  --adv_gaussian_sigma "1.25" \
  --adv_gaussian_extra_cells "2" \
  --adv_aug_k "8" \
  --degradation_size "16384" \
  --auto_resume \
  "$@"
