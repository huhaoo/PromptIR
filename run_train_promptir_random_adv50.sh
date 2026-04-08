#!/usr/bin/env bash
set -euo pipefail

cd /home/huhao/adv_ir/PromptIR

# optional quick test: append --degradation_size "128" --epochs "1"
conda run -n "promptir" --no-capture-output \
  python /home/huhao/adv_ir/source/promptir_static_adv_training.py \
  --de_type dehaze \
  --epochs "128" \
  --batch_size "4" \
  --accumulate_grad_batches "2" \
  --lr "2e-4" \
  --patch_size "128" \
  --num_workers "8" \
  --num_gpus "3" \
  --ckpt_dir "train_ckpt_random1024_adv_haze" \
  --adv_ratio "0.5" \
  --adv_cache_root "/home/huhao/adv_ir/dataset_ours/random_adv_haze" \
  --degradation_size 16384 \
  --auto_resume
