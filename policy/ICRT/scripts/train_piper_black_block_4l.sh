#!/usr/bin/env bash
set -euo pipefail

ICRT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ICRT_ROOT}"

python scripts/train.py \
  --dataset-cfg.dataset-json config/piper_black_block_lerobot_v3.json \
  --logging-cfg.output-dir output \
  --logging-cfg.log-name piper_black_block_trimmed_icrt_4l \
  --logging-cfg.wandb-project icrt-piper \
  --model-cfg.vision-encoder-cfg.vision-encoder pretrained/vision_encoder/cross-mae-rtx-vitb.pth \
  --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer_4l.json \
  --model-cfg.policy-cfg.phase pretrain \
  --model-cfg.policy-cfg.no-prompt-loss \
  --dataset-cfg.num-repeat-traj 1 \
  --dataset-cfg.non-overlapping 32 \
  --dataset-cfg.shuffle-repeat-traj \
  --shared-cfg.batch-size 1 \
  --shared-cfg.seq-length 512 \
  --shared-cfg.num-pred-steps 16 \
  --trainer-cfg.num-workers 0 \
  --trainer-cfg.no-pin-memory \
  --trainer-cfg.accum-iter 8 \
  --trainer-cfg.max-train-steps 4000 \
  --trainer-cfg.validation-every-steps 200 \
  --trainer-cfg.save-every-steps 250 \
  --trainer-cfg.log-every-steps 1 \
  --optimizer-cfg.warmup-steps 200 \
  --optimizer-cfg.lr 5e-4
