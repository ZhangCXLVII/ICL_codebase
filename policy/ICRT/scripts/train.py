from pathlib import Path
import dataclasses
import datetime
import json
import math
import numpy as np
import os
import resource
import time
import tyro 
import wandb
import yaml

import torch
import torch.backends.cudnn as cudnn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

import timm
from timm.data.loader import MultiEpochsDataLoader
from icrt.data.dataset import SequenceDataset

import icrt.util.misc as misc
from icrt.util import lr_sched
from icrt.util.misc import NativeScalerWithGradNormCount as NativeScaler
from icrt.util.args import ExperimentConfig
from icrt.util.engine import train_one_epoch
from icrt.util.model_constructor import model_constructor


def _step_loader_kwargs(args):
    kwargs = {
        "batch_size": args.shared_cfg.batch_size,
        "num_workers": args.trainer_cfg.num_workers,
        "pin_memory": args.trainer_cfg.pin_memory,
    }
    if args.trainer_cfg.num_workers > 0:
        # A sample contains 512 frames from two cameras. PyTorch's default
        # prefetch factor of two can keep several ~600 MiB samples resident.
        kwargs["prefetch_factor"] = 1
        kwargs["persistent_workers"] = False
    return kwargs


def _make_step_train_dataloader(args, dataset_train, cycle, num_tasks, global_rank):
    """Reshuffle demonstrations and create one finite training data cycle."""
    dataset_train.shuffle_dataset(cycle)
    sampler_train = misc.DistributedSubEpochSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, split_epoch=1, shuffle=True
    )
    sampler_train.set_epoch(cycle)
    return torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        drop_last=True,
        **_step_loader_kwargs(args),
    )


def _make_step_val_dataloader(args, dataset_val, cycle, num_tasks, global_rank):
    """Create validation workers only when validation is actually requested."""
    dataset_val.shuffle_dataset(cycle)
    sampler_val = misc.DistributedSubEpochSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, split_epoch=1, shuffle=False
    )
    sampler_val.set_epoch(cycle)
    if len(sampler_val) >= args.shared_cfg.batch_size:
        return torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            drop_last=False,
            **_step_loader_kwargs(args),
        )
    return None


@torch.no_grad()
def _validate_at_step(model, data_loader, device):
    if data_loader is None:
        return {}
    model.eval()
    totals = {}
    count = 0
    for dataset_item in data_loader:
        for key, value in dataset_item.items():
            dataset_item[key] = value.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = model(dataset_item)
        values = {"loss": loss.item(), **{
            key: value.item() if isinstance(value, torch.Tensor) else float(value)
            for key, value in loss_dict.items()
        }}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    model.train()
    return {key: misc.all_reduce_mean(value / max(count, 1)) for key, value in totals.items()}


def _train_by_steps(
    args,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    dataset_train,
    dataset_val,
    device,
    log_writer,
    num_tasks,
    global_rank,
):
    max_steps = args.trainer_cfg.max_train_steps
    global_step = args.shared_cfg.start_step
    accumulation = args.trainer_cfg.accum_iter
    micro_step = 0
    cycle = 0
    optimizer.zero_grad()
    model.train()
    running = {}

    print(f"Start step-based training at step {global_step}; target {max_steps} optimizer steps")
    while global_step < max_steps:
        train_loader = _make_step_train_dataloader(
            args, dataset_train, cycle, num_tasks, global_rank
        )
        if len(train_loader) == 0:
            raise RuntimeError("Training dataloader is empty")
        print(f"Data cycle {cycle}: {len(train_loader)} micro-batches")

        for dataset_item in train_loader:
            if micro_step % accumulation == 0:
                lr = lr_sched.adjust_learning_rate_step(optimizer, global_step, args)

            for key, value in dataset_item.items():
                dataset_item[key] = value.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, loss_dict = model(dataset_item)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite loss at global step {global_step}: {loss_value}")
            values = {"loss": loss_value, **{
                key: value.item() if isinstance(value, torch.Tensor) else float(value)
                for key, value in loss_dict.items()
            }}
            for key, value in values.items():
                running[key] = running.get(key, 0.0) + value

            update_grad = (micro_step + 1) % accumulation == 0
            loss_scaler(
                loss / accumulation,
                optimizer,
                parameters=model.parameters(),
                update_grad=update_grad,
            )
            micro_step += 1
            if not update_grad:
                continue

            optimizer.zero_grad()
            torch.cuda.synchronize()
            global_step += 1
            train_stats = {
                key: misc.all_reduce_mean(value / accumulation)
                for key, value in running.items()
            }
            running = {}

            if global_step % args.trainer_cfg.log_every_steps == 0:
                payload = {f"train/{key}": value for key, value in train_stats.items()}
                payload.update({
                    "train/lr": lr,
                    "system/process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
                    "system/cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024 ** 3),
                    "system/cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024 ** 3),
                    "global_step": global_step,
                })
                if misc.is_main_process():
                    if wandb.run is not None:
                        wandb.log(payload, step=global_step)
                    if log_writer is not None:
                        for key, value in payload.items():
                            if key != "global_step":
                                log_writer.add_scalar(key, value, global_step)
                    with open(os.path.join(args.logging_cfg.output_dir, "log.txt"), "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(payload) + "\n")
                print(f"step {global_step}/{max_steps} loss={train_stats['loss']:.6f} lr={lr:.3e}")

            should_validate = (
                global_step % args.trainer_cfg.validation_every_steps == 0 or global_step == max_steps
            )
            if should_validate:
                val_loader = _make_step_val_dataloader(
                    args, dataset_val, cycle, num_tasks, global_rank
                )
                val_stats = _validate_at_step(model, val_loader, device)
                del val_loader
                if val_stats and misc.is_main_process():
                    payload = {f"val/{key}": value for key, value in val_stats.items()}
                    payload["global_step"] = global_step
                    if wandb.run is not None:
                        wandb.log(payload, step=global_step)
                    if log_writer is not None:
                        for key, value in payload.items():
                            if key != "global_step":
                                log_writer.add_scalar(key, value, global_step)
                    print(f"validation step {global_step}: {val_stats}")

            should_save = global_step % args.trainer_cfg.save_every_steps == 0 or global_step == max_steps
            if args.logging_cfg.output_dir and should_save:
                misc.save_model(
                    args=args,
                    global_step=global_step,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                )

            if global_step >= max_steps:
                if log_writer is not None:
                    log_writer.flush()
                return
        cycle += 1

def main(args : ExperimentConfig):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.shared_cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Loading data config
    data_cfg = json.load(open(args.dataset_cfg.dataset_json, 'r'))

    # make sure the number of cameras is correct 
    rgb_observations = data_cfg["image_keys"]
    assert len(rgb_observations) == args.shared_cfg.num_cameras, "Number of cameras must match the number of rgb observations"

    model = model_constructor(
        model_config=args.model_cfg, 
        shared_config=args.shared_cfg,
        train=args.train,
    )

    timm_data_cfg = timm.data.resolve_data_config(model.vision_encoder.model.pretrained_cfg)
    no_aug_vision_transform = timm.data.create_transform(**timm_data_cfg)
    if args.dataset_cfg.vision_aug:
        timm_data_cfg["is_training"] = True
        timm_data_cfg["hflip"] = 0.0
        timm_data_cfg["scale"] = (0.65, 1.0)
        timm_data_cfg["ratio"] = (1.0, 1.0)
    vision_transform = timm.data.create_transform(**timm_data_cfg)

    model.to(device)

    model_without_ddp = model
    
    # controlled by --model-cfg.policy-cfg.pretrained_path flag
    if args.model_cfg.policy_cfg.pretrained_path is not None: 
        print("Finetuning from %s" % args.model_cfg.policy_cfg.pretrained_path)
        misc.load_model(model_without_ddp, args.model_cfg.policy_cfg.pretrained_path)
    
    print("Model trainable params: ")
    print(model_without_ddp.state_dict().keys())

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # training detail
    eff_batch_size = args.shared_cfg.batch_size * args.trainer_cfg.accum_iter * misc.get_world_size()

    if args.optimizer_cfg.lr is None:  # only base_lr is specified
        args.optimizer_cfg.lr = args.optimizer_cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.optimizer_cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.optimizer_cfg.lr)

    print("accumulate grad iterations: %d" % args.trainer_cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.optimizer_cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.optimizer_cfg.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    total, trainable = model_without_ddp.get_total_parameters(), model_without_ddp.get_trainable_parameters()
    print("trainable: ", trainable)
    print("Total params: ", total)
    print("percentage trainable: ", trainable / total)
    # --resume
    misc.resume_from_ckpt(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
        
    dataset_train = SequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        vision_transform=vision_transform,
        no_aug_vision_transform=no_aug_vision_transform,
        split="train",
    )
    val_dataset_config = dataclasses.replace(
        args.dataset_cfg,
        vision_aug=False,
        proprio_noise=0.0,
        action_noise=0.0,
    )
    dataset_val = SequenceDataset(
        dataset_config=val_dataset_config,
        shared_config=args.shared_cfg,
        vision_transform=no_aug_vision_transform,
        no_aug_vision_transform=no_aug_vision_transform,
        split="val"
    )
    print("Length of dataset_train: ", len(dataset_train))
    print("Length of dataset_val: ", len(dataset_val))
    
    # save the train the val splits 
    dataset_train.save_split(os.path.join(args.logging_cfg.output_dir, "train_split.json"))
    dataset_val.save_split(os.path.join(args.logging_cfg.output_dir, "val_split.json"))

    # Step mode logs directly to W&B; legacy epoch mode can mirror TensorBoard.
    if global_rank == 0 and args.logging_cfg.log_name is not None:
        wandb.init(
            entity=args.logging_cfg.wandb_entity,
            project=args.logging_cfg.wandb_project,
            config=args,
            name=args.logging_cfg.log_name,
            sync_tensorboard=args.trainer_cfg.max_train_steps is None and SummaryWriter is not None,
        )
        if args.trainer_cfg.max_train_steps is not None:
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("val/*", step_metric="global_step")

    # SummaryWrite
    if args.trainer_cfg.max_train_steps is not None:
        # Step mode writes directly to W&B and JSONL with global_step.
        log_writer = None
    elif global_rank == 0 and args.logging_cfg.log_dir is not None and SummaryWriter is not None:
        os.makedirs(args.logging_cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.logging_cfg.log_dir)
    else:
        log_writer = None

    if args.trainer_cfg.max_train_steps is not None:
        start_time = time.time()
        _train_by_steps(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            device=device,
            log_writer=log_writer,
            num_tasks=num_tasks,
            global_rank=global_rank,
        )
        total_time = time.time() - start_time
        print("Training time {}".format(datetime.timedelta(seconds=int(total_time))))
        return

    if args.trainer_cfg.epochs is None:
        raise ValueError("Set either --trainer-cfg.max-train-steps or --trainer-cfg.epochs")
    print(f"Start training for {args.trainer_cfg.epochs} epochs")
    start_time = time.time()

    # for resume, we need to instantiate new samplers 
    resume_reload = args.shared_cfg.resume is not None

    for epoch in range(args.shared_cfg.start_epoch, args.trainer_cfg.epochs):

        if resume_reload or epoch % args.shared_cfg.split_epoch == 0:
            print(f"Shuffling sequences every {args.shared_cfg.split_epoch} epochs, epoch: {epoch}")
            dataset_train.shuffle_dataset(epoch)
            dataset_val.shuffle_dataset(epoch)
            print("Recreating dataloaders ...")
            sampler_train = misc.DistributedSubEpochSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, split_epoch=args.shared_cfg.split_epoch, shuffle=True
            )
            sampler_val = misc.DistributedSubEpochSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, split_epoch=args.shared_cfg.split_epoch, shuffle=False
            )
            print("Sampler_train = %s" % str(sampler_train))
            print("length of train sampler: ", len(sampler_train))
            print("Sampler_val = %s" % str(sampler_val))
            print("length of val sampler: ", len(sampler_val))
            data_loader_train = MultiEpochsDataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.shared_cfg.batch_size,
                num_workers=args.trainer_cfg.num_workers,
                pin_memory=args.trainer_cfg.pin_memory,
                drop_last=True,
            )
            if len(sampler_val) > args.shared_cfg.batch_size:
                data_loader_val = MultiEpochsDataLoader(
                    dataset_val, sampler=sampler_val,
                    batch_size=args.shared_cfg.batch_size,
                    num_workers=args.trainer_cfg.num_workers,
                    pin_memory=args.trainer_cfg.pin_memory,
                    drop_last=True,
                )
            else:
                data_loader_val = None
            print("Done recreating dataloaders!")
            print("length of dataset: ", len(dataset_train))
            print("length of dataloader: ", len(data_loader_train))
            resume_reload = False

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            if data_loader_val is not None:
                data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if data_loader_val is not None:
            with torch.no_grad():
                val_stats = train_one_epoch(
                    model, data_loader_val,
                    optimizer, device, epoch, loss_scaler,
                    log_writer=log_writer, validate=True,
                    args=args
                )

            print("Validation Epoch {}".format(epoch))
            
        if args.logging_cfg.output_dir and (epoch % args.shared_cfg.save_every == 0 or epoch + 1 == args.trainer_cfg.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if data_loader_val is not None:
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})

        if args.logging_cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.logging_cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    # parsing args 
    args = tyro.cli(ExperimentConfig)

    if args.load_config is not None: 
        print("loading configs from file: ", args.load_config)
        assert os.path.exists(args.load_config), f"Config file does not exist: {args.load_config}"
        args : ExperimentConfig = yaml.load(Path(args.load_config).read_text(), Loader=yaml.Loader) 

    # creating the output directory and logging directory 
    if args.logging_cfg.log_name is not None: 
        args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, args.logging_cfg.log_name)
    if args.logging_cfg.log_dir is None:
        args.logging_cfg.log_dir = args.logging_cfg.output_dir
    if args.logging_cfg.output_dir:
        Path(args.logging_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # dump the args into a yaml file 
    with open(os.path.join(args.logging_cfg.output_dir, "run.yaml"), 'w') as f:
        yaml.dump(args, f)

    main(args)
