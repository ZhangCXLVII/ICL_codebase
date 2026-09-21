# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from icrt.util.args import ExperimentConfig

def adjust_learning_rate(optimizer, epoch : int, args : ExperimentConfig):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.optimizer_cfg.warmup_epochs:
        lr = args.optimizer_cfg.lr * epoch / args.optimizer_cfg.warmup_epochs 
    else:
        lr = args.optimizer_cfg.min_lr + (args.optimizer_cfg.lr - args.optimizer_cfg.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.optimizer_cfg.warmup_epochs) / (args.trainer_cfg.epochs - args.optimizer_cfg.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_step(optimizer, step: int, args: ExperimentConfig):
    """Warm up and cosine-decay by optimizer-update step."""
    total_steps = args.trainer_cfg.max_train_steps
    warmup_steps = min(args.optimizer_cfg.warmup_steps, total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        lr = args.optimizer_cfg.lr * (step + 1) / warmup_steps
    else:
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        lr = args.optimizer_cfg.min_lr + (args.optimizer_cfg.lr - args.optimizer_cfg.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
    return lr
