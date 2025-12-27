import argparse
from datetime import datetime
import os
from pprint import pformat
from typing import Any, Dict, Optional
import warnings

import torch
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.RecDataModule import DataConfig, RecDataModule
from src.models.LightGCNModule import LightGCNModule
from src.utils.seed import seed_everything_strict


torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train LightGCN with RecDataModule (YAML optional, CLI overrides).")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--processed-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--num-negatives", type=int, default=None)

    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-fold", type=int, default=None)
    p.add_argument("--dropout-flag", action="store_true")
    p.add_argument("--dropout-rate", type=float, default=None)
    p.add_argument("--reg-ratio", type=float, default=None)
    p.add_argument("--topk", type=str, default=None, help='e.g. "10,20"')

    p.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ckpt-root", type=str, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--monitor", type=str, default=None)
    p.add_argument("--early-stop-delta", type=float, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--save-epoch-interval", type=int, default=None)
    p.add_argument("--val-epoch-interval", type=int, default=None)
    p.add_argument("--accelerator", type=str, default=None)
    p.add_argument("--devices", type=str, default=None)

    return p.parse_args()


def load_maybe_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".yml", ".yaml"]:
        raise ValueError(f"--config must be a YAML file, got {ext}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict.")
    return cfg


def parse_topk(s: Optional[str]) -> Optional[list[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_cli_override(args: argparse.Namespace) -> dict:
    return {
        "data": {
            "dataset": args.dataset,
            "processed_dir": args.processed_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_negatives": args.num_negatives,
        },
        "model": {
            "model_name": args.model_name,
            "embed_dim": args.embed_dim,
            "num_layers": args.num_layers,
            "num_fold": args.num_fold,
            "dropout_flag": True if args.dropout_flag else None,
            "dropout_rate": args.dropout_rate,
            "reg_ratio": args.reg_ratio,
            "topk": parse_topk(args.topk),
        },
        "optim": {
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "train": {
            "seed": args.seed,
            "ckpt_root": args.ckpt_root,
            "max_epochs": args.max_epochs,
            "monitor": args.monitor,
            "early_stop_delta": args.early_stop_delta,
            "early_stop_patience": args.early_stop_patience,
            "save_epoch_interval": args.save_epoch_interval,
            "val_epoch_interval": args.val_epoch_interval,
            "accelerator": args.accelerator,
            "devices": args.devices,
        },
    }


def merge_yaml_with_cli(yaml_cfg: dict, cli_cfg: dict) -> dict:
    merged = {
        "data": dict(yaml_cfg.get("data") or {}),
        "model": dict(yaml_cfg.get("model") or {}),
        "optim": dict(yaml_cfg.get("optim") or {}),
        "train": dict(yaml_cfg.get("train") or {}),
    }

    for k, v in cli_cfg.get("data", {}).items():
        if v is not None:
            merged["data"][k] = v

    for k, v in cli_cfg.get("model", {}).items():
        if v is not None:
            merged["model"][k] = v

    for k, v in cli_cfg.get("optim", {}).items():
        if v is not None:
            merged["optim"][k] = v

    for k, v in cli_cfg.get("train", {}).items():
        if v is not None:
            merged["train"][k] = v

    return merged


def validate_cfg(cfg: Dict[str, Any]):
    # ----- data config -----
    data_cfg_dict = cfg.get("data", {})
    dataset = data_cfg_dict.get("dataset")
    processed_dir = data_cfg_dict.get("processed_dir")
    if not processed_dir or not dataset:
        raise ValueError("Missing data.dataset or data.processed_dir (YAML or CLI)")

    data_cfg = DataConfig(
        dataset=dataset,
        processed_dir=processed_dir,
        batch_size=int(data_cfg_dict.get("batch_size", 2048)),
        num_workers=int(data_cfg_dict.get("num_workers", 4)),
        num_negatives=int(data_cfg_dict.get("num_negatives", 1)),
    )

    # ----- model config -----
    model_cfg_dict = cfg.get("model", {})
    topk = model_cfg_dict.get("topk", [10, 20])
    model_cfg = {
        "model_name": data_cfg_dict.get("model_name", "lightgcn"),
        "embed_dim": int(model_cfg_dict.get("embed_dim", 64)),
        "num_layers": int(model_cfg_dict.get("num_layers", 3)),
        "num_fold": int(model_cfg_dict.get("num_fold", 100)),
        "dropout_flag": bool(model_cfg_dict.get("dropout_flag", False)),
        "dropout_rate": float(model_cfg_dict.get("dropout_rate", 0.1)),
        "reg_ratio": float(model_cfg_dict.get("reg_ratio", 0.001)),
        "topk": topk,
    }

    # ----- optim config -----
    optim_cfg_dict = cfg.get("optim", {})
    optim_cfg = {
        "optimizer": str(optim_cfg_dict.get("optimizer", "adam")),
        "lr": float(optim_cfg_dict.get("lr", 1e-3)),
        "weight_decay": float(optim_cfg_dict.get("weight_decay", 1e-6)),
    }

    # ----- train config -----
    train_cfg_dict = cfg.get("train", {})

    train_cfg = {
        "seed": int(train_cfg_dict.get("seed", 42)),
        "save_root": str(train_cfg_dict.get("save_root", "./ckpts")),
        "max_epochs": int(train_cfg_dict.get("max_epochs", 200)),
        "monitor": str(train_cfg_dict.get("monitor", "val/Recall@20")),
        "early_stop_delta": float(train_cfg_dict.get("early_stop_delta", 0.0)),
        "early_stop_patience": int(train_cfg_dict.get("early_stop_patience", 3)),
        "save_epoch_interval":  int(train_cfg_dict.get("save_epoch_interval", 100)),
        "val_epoch_interval": int(train_cfg_dict.get("val_epoch_interval", 20)),
        "accelerator": str(train_cfg_dict.get("accelerator", "auto")),
        "devices": str(train_cfg_dict.get("devices", "auto")),
    }

    return data_cfg, model_cfg, optim_cfg, train_cfg


def pretty_print_cfg(
    data_cfg,
    model_cfg: Dict[str, Any],
    optim_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
):
    cfg_dict = {
        "data": vars(data_cfg) if hasattr(data_cfg, "__dict__") else data_cfg,
        "model": model_cfg,
        "optim": optim_cfg,
        "train": train_cfg,
    }

    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(pformat(cfg_dict, indent=2, width=100, sort_dicts=False))
    print("=" * 80)


def train(cfg: Dict[str, Any]) -> None:
    data_cfg, model_cfg, optim_cfg, train_cfg = validate_cfg(cfg)
    pretty_print_cfg(data_cfg, model_cfg, optim_cfg, train_cfg)

    # ----- initial -----
    seed = train_cfg.get("seed")
    seed_everything_strict(seed)

    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(train_cfg.get("save_root"), f"train_{model_cfg["model_name"]}_{data_cfg.dataset}_{time_str}")

    # ----- datamodule -----
    dm = RecDataModule(cfg=data_cfg, seed=seed)
    dm.setup("fit")

    # ----- lightning module -----
    val_gt, test_gt = dm.get_gt()
    model = LightGCNModule(
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        num_users=int(dm.num_users),
        num_items=int(dm.num_items),
        norm_adj=dm.get_norm_adj(),
        train_interactions=dm.get_train_interactions(),
        val_ground_truth=val_gt,
        test_ground_truth=test_gt,
    )

    # ----- trainer -----
    early_stop = EarlyStopping(
        monitor=train_cfg.get("monitor"),
        mode="max",
        min_delta=train_cfg.get("early_stop_delta"),
        patience=train_cfg.get("early_stop_patience"),
    )

    periodic_ckpt = ModelCheckpoint(
        dirpath=save_path,
        filename="epoch{epoch:04d}-step{step:04d}",
        every_n_epochs=train_cfg.get("save_epoch_interval"),
        save_top_k=-1,
        save_last=True,
    )

    best_ckpt = ModelCheckpoint(
        dirpath=save_path,
        filename="best-epoch{epoch:04d}-step{step:04d}",
        monitor=train_cfg.get("monitor"),
        mode="max",
        save_top_k=1,
        save_last=False,
    )

    logger = TensorBoardLogger(save_dir=os.path.join(save_path, "logs"), name="tb")
    
    trainer = pl.Trainer(
        callbacks=[early_stop, periodic_ckpt, best_ckpt],
        logger=logger,
        enable_checkpointing=True,
        max_epochs=train_cfg.get("max_epochs"),
        check_val_every_n_epoch=train_cfg.get("val_epoch_interval"),
        precision=32,
        accelerator=train_cfg.get("accelerator"),
        devices=train_cfg.get("devices"),
    )

    dummy_val_loader = torch.utils.data.DataLoader([0], batch_size=1)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dummy_val_loader)
    trainer.test(model, dataloaders=dummy_val_loader, ckpt_path="best")


def main():
    args = parse_args()

    yaml_cfg = load_maybe_config(args.config)
    cli_cfg = build_cli_override(args)
    merged = merge_yaml_with_cli(yaml_cfg, cli_cfg)

    train(merged)


if __name__ == "__main__":
    main()
