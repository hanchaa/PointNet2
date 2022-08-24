import os
import copy
import importlib

import torch
from torch.cuda import device_count
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, all_reduce
from tqdm import tqdm
import wandb

from utils import parse_args, get_logger, is_mainprocess, get_lr
from data_utils import build_dataset


def test(model, args):
    assert args.test_batch_size % args.num_gpus == 0, "Total test batch size must be a multiple of the number of gpus"
    dataset = build_dataset(args, "test")
    dataloader = DataLoader(
        dataset,
        batch_size=args.test_batch_size // args.num_gpus,
        pin_memory=True,
        num_workers=8,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=False)
    )

    with torch.no_grad():
        model = model.eval()
        correct = 0

        for points, target in (tqdm(dataloader) if is_mainprocess() else dataloader):
            points, target = points.cuda(), target.cuda()
            pred = model(points)
            pred = pred.argmax(dim=-1)
            correct += pred.eq(target).sum()

        correct = correct.unsqueeze(0)
        all_reduce(correct)

        acc = correct[0] / len(dataset)

        return acc


def train(model, args, logger, loss_fn):
    assert args.batch_size % args.num_gpus == 0, "Total batch size must be a multiple of the number of gpus"
    dataset = build_dataset(args, "train")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.num_gpus,
        pin_memory=True,
        num_workers=8,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

    if is_mainprocess():
        logger.info(f"Loaded {len(dataset)} points")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    try:
        checkpoint = torch.load(args.model_weights)
        model.load_state_dict(checkpoint["model"])

        if args.resume:
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

        if is_mainprocess():
            logger.info(f"Model weights are loaded from {args.model_weights}")
    except:
        if is_mainprocess():
            logger.info(f"No existing model, generating random seeds")

    if is_mainprocess():
        logger.info(f"Starting training from epoch {start_epoch + 1}")

    best_val_acc = 0
    best_model = None
    best_optimizer = None
    best_lr_scheduler = None

    for epoch in range(start_epoch, args.epoch):
        if is_mainprocess():
            logger.info(f"Epoch {epoch + 1} / {args.epoch}")

        model = model.train()

        for points, target in (tqdm(dataloader) if is_mainprocess() else dataloader):
            optimizer.zero_grad()
            points, target = points.cuda(), target.cuda()

            pred = model(points)
            loss = loss_fn(pred, target)

            loss.backward()
            optimizer.step()

            pred = F.softmax(pred, dim=-1).argmax(dim=-1)
            correct = pred.eq(target).sum() / points.shape[0]

            train_result = torch.stack([loss, correct])
            all_reduce(train_result)
            train_result = train_result / args.num_gpus

            if is_mainprocess() and args.wandb != "":
                wandb.log({"loss": train_result[0], "train acc": train_result[1] * 100, "lr": get_lr(optimizer)})

        scheduler.step()

        val_acc = test(model, args, logger)

        if is_mainprocess():
            wandb.log({"val acc": val_acc * 100})

            logger.info(f"Train loss: {train_result[0]:.4f} / Train accuracy: {train_result[1] * 100:.2f}%")
            logger.info(f"Validation accuracy: {val_acc * 100:.2f}%")
            logger.info(f"lr: {get_lr(optimizer)}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
                best_optimizer = copy.deepcopy(optimizer)
                best_lr_scheduler = copy.deepcopy(scheduler)

            if (epoch + 1) % args.checkpoint_period == 0:
                logger.info(f"Saving checkpoint model_{epoch}.pth")
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }, os.path.join(args.output_dir, f"model_{epoch}.pth"))

    if is_mainprocess():
        logger.info("Saving best model to model_best.pth")
        torch.save({
            "epoch": epoch,
            "model": best_model.state_dict(),
            "optimizer": best_optimizer.state_dict(),
            "scheduler": best_lr_scheduler.state_dict()
        }, os.path.join(args.output_dir, f"model_best.pth"))


def main(device, args):
    init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:3456",
        world_size=args.num_gpus,
        rank=device
    )
    torch.cuda.set_device(device)

    module = importlib.import_module(f"modeling.architectures.{args.model}")
    model = module.get_model(args).cuda()
    model = DistributedDataParallel(model, device_ids=[device])
    loss_fn = module.get_loss_fn()

    logger = None
    if is_mainprocess():
        logger = get_logger(args.output_dir)
        logger.info(f"Parameters: {args}")

        if args.wandb != "":
            wandb.init(entity=args.wandb, project="pointnet2", resume=args.resume, id=args.wandb_run_id)
            wandb.watch(model, log="all")

    if args.eval_only:
        acc = test(model, args, logger)

        if is_mainprocess() and args.wandb != "":
            wandb.log({"val acc": acc * 100})
            logger.info(f"Validation accuracy: {acc:.2f}%")
        return

    train(model, args, logger, loss_fn)


if __name__ == "__main__":
    args = parse_args()
    assert args.num_gpus <= device_count(), f"The number of gpus should be less than or equal to {device_count()}"
    mp.spawn(main, nprocs=args.num_gpus, args=(args, ))