# optional quick test: append --degradation_size "128"
conda run -n "promptir" --no-capture-output \
  python train_expairs.py \
  --de_type denoise_50 \
  --epochs "128" \
  --batch_size "3" \
  --accumulate_grad_batches "3" \
  --lr "2e-4" \
  --patch_size "128" \
  --num_workers "8" \
  --num_gpus "3" \
  --p_target_m_target "0.1" \
  --p_target_mm_input "0.1" \
  --ckpt_dir "/home/huhao/adv_ir/exp/train_ckpt_noise50_expairs" \
  --degradation_size 32768 \
  --auto_resume