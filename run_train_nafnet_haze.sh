#!/usr/bin/env bash
set -euo pipefail

cd /home/huhao/adv_ir/PromptIR

conda run -n promptir --no-capture-output \
  python train.py \
  --model_arch "nafnet" \
  --de_type dehaze \
  --epochs "128" \
  --batch_size "4" \
  --accumulate_grad_batches "2" \
  --lr "2e-4" \
  --patch_size "128" \
  --num_workers "8" \
  --num_gpus "3" \
  --ckpt_dir "/home/huhao/adv_ir/exp/train_ckpt_nafnet_haze" \
  --degradation_size "16384" \
  --auto_resume
