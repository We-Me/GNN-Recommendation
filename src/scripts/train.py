import argparse
from datetime import datetime
import os
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.RecDataModule import DataConfig, RecDataModule
from src.models.LightGCNModule import LightGCNModule
from src.utils.seed import seed_everything_strict

torch.set_float32_matmul_precision("high")


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


def build_cli_override(args: argparse.Namespace) -> dict:
    return {
        "seed": args.seed,
        "data": {
            "processed_dir": args.processed_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_negatives": args.num_negatives,
        },
        "model": {
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
        "trainer": {
            "max_epochs": args.max_epochs,
            "check_val_every_n_epoch": args.check_val_every_n_epoch,
            "every_n_epochs": args.every_n_epochs,
            "monitor": args.monitor,
            "precision": args.precision,
            "log_every_n_steps": args.log_every_n_steps,
            "ckpt_root": args.ckpt_root,
            "accelerator": args.accelerator,
            "devices": args.devices,
        },
    }


def merge_yaml_with_cli(yaml_cfg: dict, cli_cfg: dict) -> dict:
    merged = {
        "seed": yaml_cfg.get("seed"),
        "data": dict(yaml_cfg.get("data") or {}),
        "model": dict(yaml_cfg.get("model") or {}),
        "optim": dict(yaml_cfg.get("optim") or {}),
        "trainer": dict(yaml_cfg.get("trainer") or {}),
    }

    if cli_cfg.get("seed") is not None:
        merged["seed"] = cli_cfg["seed"]

    for k, v in cli_cfg.get("data", {}).items():
        if v is not None:
            merged["data"][k] = v

    for k, v in cli_cfg.get("model", {}).items():
        if v is not None:
            merged["model"][k] = v

    for k, v in cli_cfg.get("optim", {}).items():
        if v is not None:
            merged["optim"][k] = v

    for k, v in cli_cfg.get("trainer", {}).items():
        if v is not None:
            merged["trainer"][k] = v

    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train LightGCN with RecDataModule (YAML optional, CLI overrides).")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config path")

    p.add_argument("--seed", type=int, default=None)
    
    p.add_argument("--processed-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--num-negatives", type=int, default=None)

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

    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--monitor", type=str, default=None)
    p.add_argument("--check-val-every-n-epoch", type=int, default=None)
    p.add_argument("--every-n-epochs", type=int, default=None)
    p.add_argument("--precision", type=str, default=None, help='e.g. "32-true", "16-mixed"')
    p.add_argument("--log-every-n-steps", type=int, default=None)
    p.add_argument("--ckpt-root", type=str, default=None)
    p.add_argument("--accelerator", type=str, default=None)
    p.add_argument("--devices", type=str, default=None)

    return p.parse_args()


def parse_topk(s: Optional[str]) -> Optional[list[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def train(cfg: Dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 42))
    seed_everything_strict(seed)

    # ----- datamodule -----
    data_cfg_dict = cfg.get("data", {})
    processed_dir = data_cfg_dict.get("processed_dir")
    if not processed_dir:
        raise ValueError("Missing data.processed_dir (YAML or --processed-dir)")

    data_cfg = DataConfig(
        processed_dir=processed_dir,
        batch_size=int(data_cfg_dict.get("batch_size", 2048)),
        num_workers=int(data_cfg_dict.get("num_workers", 4)),
        num_negatives=int(data_cfg_dict.get("num_negatives", 1)),
    )
    dm = RecDataModule(cfg=data_cfg, seed=seed)
    dm.setup("fit")

    # ----- lightning module -----
    model_cfg_dict = cfg.get("model", {})
    topk = model_cfg_dict.get("topk", [10, 20])
    model_cfg = {
        "embed_dim": int(model_cfg_dict.get("embed_dim", 64)),
        "num_layers": int(model_cfg_dict.get("num_layers", 3)),
        "num_fold": int(model_cfg_dict.get("num_fold", 100)),
        "dropout_flag": bool(model_cfg_dict.get("dropout_flag", False)),
        "dropout_rate": float(model_cfg_dict.get("dropout_rate", 0.1)),
        "reg_ratio": float(model_cfg_dict.get("reg_ratio", 1.0)),
        "topk": topk,
    }

    optim_cfg_dict = cfg.get("optim", {})
    optim_cfg = {
        "optimizer": str(optim_cfg_dict.get("optimizer", "adam")),
        "optim_params": {
            "lr": float(optim_cfg_dict.get("lr", 1e-3)),
            "weight_decay": float(optim_cfg_dict.get("weight_decay", 1e-6)),
        },
    }

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
    trainer_cfg = cfg.get("trainer", {})
    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(trainer_cfg.get("save_root", "./ckpts"), f"train_{time_str}")
    os.makedirs(save_path, exist_ok=True)

    monitor = str(trainer_cfg.get("monitor", "val/Recall@20"))
    every_n_epochs = int(trainer_cfg.get("every_n_epochs", 10))

    best_ckpt = ModelCheckpoint(
        dirpath=save_path,
        filename="best-epoch{epoch:04d}-step{step:08d}",
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=False,
    )

    periodic_ckpt = ModelCheckpoint(
        dirpath=save_path,
        filename="epoch{epoch:04d}-step{step:08d}",
        every_n_epochs=every_n_epochs,
        save_top_k=-1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor=monitor,
        mode="max",
        patience=3,
        min_delta=0.001
    )

    logger = TensorBoardLogger(save_dir=os.path.join(save_path, "logs"), name="tb")
    
    trainer = pl.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 200)),
        check_val_every_n_epoch=int(trainer_cfg.get("check_val_every_n_epoch", 5)),
        callbacks=[best_ckpt, periodic_ckpt, early_stop],
        logger=logger,
        enable_checkpointing=True,
        accelerator=str(trainer_cfg.get("accelerator", "auto")),
        devices=trainer_cfg.get("devices", "auto"),
        precision=str(trainer_cfg.get("precision", "32-true")),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
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
