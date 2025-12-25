import argparse
import os
from typing import Optional

from data.preprocess import RatingPreprocessor, RawConfig, SplitConfig

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise RuntimeError("This script requires PyYAML: pip install pyyaml") from e


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
        "processed_dir": args.processed_dir,
        "raw": {
            "path": args.path,
            "sep": args.sep,
            "has_header": args.has_header if args.has_header else None,
            "user_col": args.user_col,
            "item_col": args.item_col,
            "rating_col": args.rating_col,
        },
        "split": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "min_user_interactions": args.min_user_interactions,
        },
    }


def merge_yaml_with_cli(yaml_cfg: dict, cli_cfg: dict) -> dict:
    merged = {
        "processed_dir": yaml_cfg.get("processed_dir"),
        "raw": dict(yaml_cfg.get("raw") or {}),
        "split": dict(yaml_cfg.get("split") or {}),
    }

    if cli_cfg.get("processed_dir") is not None:
        merged["processed_dir"] = cli_cfg["processed_dir"]

    for k, v in cli_cfg.get("raw", {}).items():
        if v is not None:
            merged["raw"][k] = v

    for k, v in cli_cfg.get("split", {}).items():
        if v is not None:
            merged["split"][k] = v

    return merged


def validate_and_instantiate(cfg: dict) -> tuple[RawConfig, SplitConfig, str]:
    raw = cfg.get("raw") or {}
    split = cfg.get("split") or {}

    raw_cfg = RawConfig(**raw) if raw else RawConfig(path="")
    split_cfg = SplitConfig(**split) if split else SplitConfig()
    processed_dir = cfg.get("processed_dir")

    if not raw_cfg.path:
        raise ValueError("Missing raw.path (CLI or YAML)")
    if not processed_dir:
        raise ValueError("Missing processed_dir (CLI or YAML)")

    return raw_cfg, split_cfg, str(processed_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Preprocess rating data (YAML optional, CLI overrides)")
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--processed-dir", type=str, default=None)

    p.add_argument("--path", type=str, default=None)
    p.add_argument("--sep", type=str, default=None)
    p.add_argument("--has-header", action="store_true")
    p.add_argument("--user-col", type=int, default=None)
    p.add_argument("--item-col", type=int, default=None)
    p.add_argument("--rating-col", type=int, default=None)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--test-ratio", type=float, default=None)
    p.add_argument("--min-user-interactions", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    yaml_cfg = load_maybe_config(args.config)
    cli_cfg = build_cli_override(args)
    merged = merge_yaml_with_cli(yaml_cfg, cli_cfg)

    raw_cfg, split_cfg, processed_dir = validate_and_instantiate(merged)

    proc = RatingPreprocessor(raw_cfg, split_cfg, processed_dir)
    meta = proc.run()


if __name__ == "__main__":
    main()
