# optional quick test: append --degradation_size "128"
conda run -n "promptir" --no-capture-output \
  python train_expairs.py \
  --de_type denoise_15 denoise_25 denoise_50 derain dehaze \
  --epochs "128" \
  --batch_size "3" \
  --accumulate_grad_batches "3" \
  --lr "2e-4" \
  --patch_size "128" \
  --num_workers "8" \
  --num_gpus "2" \
  --p_target_m_target "0" \
  --p_target_mm_input "0.2" \
  --ckpt_dir "train_ckpt" \
  --auto_resume