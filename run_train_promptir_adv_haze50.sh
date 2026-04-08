#!/usr/bin/env bash
set -euo pipefail

cd /home/huhao/adv_ir/PromptIR

conda run -n promptir --no-capture-output \
  python adv_train.py \
  --de_type dehaze \
  --epochs "128" \
  --batch_size "4" \
  --accumulate_grad_batches "2" \
  --lr "2e-4" \
  --patch_size "128" \
  --num_workers "8" \
  --num_gpus "3" \
  --ckpt_dir "train_ckpt_adv_haze50" \
  --adv_ratio "0.5" \
  --adv_resample_epochs "4" \
  --adv_steps1 "2" \
  --adv_steps2 "4" \
  --adv_step_size "0.03" \
  --adv_lambda_reg "0.05" \
  --adv_promptir_patch_size "256" \
  --adv_promptir_patch_overlap "32" \
  --adv_cache_root "/home/huhao/adv_ir/PromptIR/data/Train/adv_pairs" \
  --adv_attack_device "cuda" \
  --adv_aug_k "8" \
  --degradation_size 16384 \
  --auto_resume
