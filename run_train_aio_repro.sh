conda run -n "promptir" --no-capture-output \
  python train.py \
  --de_type denoise_15 denoise_25 denoise_50 derain dehaze \
  --epochs "64" \
  --batch_size "4" \
  --accumulate_grad_batches "2" \
  --lr "2e-4" \
  --patch_size "128" \
  --num_workers "8" \
  --num_gpus "2" \
  --ckpt_dir "train_ckpt" \
#   --degradation_size 128