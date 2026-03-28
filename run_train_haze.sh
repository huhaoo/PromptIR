# optional quick test: append --degradation_size "128"
conda run -n "promptir" --no-capture-output \
  python train.py \
  --de_type dehaze \
  --epochs "128" \
  --batch_size "4" \
  --accumulate_grad_batches "2" \
  --lr "2e-4" \
  --patch_size "128" \
  --num_workers "8" \
  --num_gpus "3" \
  --ckpt_dir "train_ckpt_dehaze" \
  --degradation_size 16384 \
  --auto_resume