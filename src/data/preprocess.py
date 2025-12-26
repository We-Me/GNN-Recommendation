import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class RawConfig:
    path: str
    sep: str
    has_header: bool
    user_col: int
    item_col: int
    rating_col: int


@dataclass
class SplitConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    min_user_interactions: int


@dataclass
class ProcessedMeta:
    raw_path: str
    num_users: int
    num_items: int
    sizes: Dict[str, int]
    raw_config: dict
    split_config: dict
    note: str


class RatingPreprocessor:
    def __init__(
        self,
        raw_cfg: RawConfig,
        split_cfg: SplitConfig,
        processed_dir: str
    ):
        self.raw_cfg = raw_cfg
        self.split_cfg = split_cfg
        self.processed_dir = processed_dir

        self.user2idx: Dict[str, int] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2user: Dict[int, str] = {}
        self.idx2item: Dict[int, str] = {}
        self.num_users: int = 0
        self.num_items: int = 0

        self.train: Optional[np.ndarray] = None
        self.val: Optional[np.ndarray] = None
        self.test: Optional[np.ndarray] = None

    def run(self) -> ProcessedMeta:
        os.makedirs(self.processed_dir, exist_ok=True)

        triplets = self._read_raw_triplets(self.raw_cfg)
        u, i, r = self._factorize_triplets(triplets)

        self.num_users = int(u.max() + 1) if u.size else 0
        self.num_items = int(i.max() + 1) if i.size else 0

        train, val, test = self._split_per_user(u, i, r, self.num_users, self.split_cfg)
        self.train, self.val, self.test = train, val, test

        np.save(os.path.join(self.processed_dir, "train.npy"), train)
        np.save(os.path.join(self.processed_dir, "val.npy"), val)
        np.save(os.path.join(self.processed_dir, "test.npy"), test)

        with open(os.path.join(self.processed_dir, "id_maps.json"), "w", encoding="utf-8") as f:
            json.dump({"user2idx": self.user2idx, "item2idx": self.item2idx, 
                       "idx2user": self.idx2user, "idx2item": self.idx2item}, 
                       f, ensure_ascii=False)

        meta = ProcessedMeta(
            raw_path=self.raw_cfg.path,
            num_users=self.num_users,
            num_items=self.num_items,
            sizes={"train": int(train.shape[0]), "val": int(val.shape[0]), "test": int(test.shape[0])},
            raw_config=asdict(self.raw_cfg),
            split_config=asdict(self.split_cfg),
            note="train/val/test store (u,i,rating) float32; BPR uses cols 0,1; rating kept for future.",
        )
        self._save_meta(meta)
        return meta

    @staticmethod
    def _read_raw_triplets(cfg: RawConfig) -> List[Tuple[str, str, float]]:
        out: List[Tuple[str, str, float]] = []
        max_parts = max(cfg.user_col, cfg.item_col, cfg.rating_col) \
            if cfg.rating_col else max(cfg.user_col, cfg.item_col)
        
        with open(cfg.path, "r", encoding="utf-8") as f:
            if cfg.has_header:
                next(f, None)

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(cfg.sep)
                if len(parts) <= max_parts:
                    continue
                u = parts[cfg.user_col]
                it = parts[cfg.item_col]

                if cfg.rating_col:
                    try:
                        r = float(parts[cfg.rating_col])
                    except ValueError:
                        continue
                else:
                    r = 1.0
                    
                out.append((u, it, r))

        return out

    def _factorize_triplets(self, triplets: List[Tuple[str, str, float]]):
        self.user2idx = {}
        self.item2idx = {}
        self.idx2user = {}
        self.idx2item = {}

        u_idx: List[int] = []
        i_idx: List[int] = []
        r_val: List[float] = []

        for u, it, r in triplets:
            if u not in self.user2idx:
                self.user2idx[u] = len(self.user2idx)
                self.idx2user[len(self.idx2user)] = u
            if it not in self.item2idx:
                self.item2idx[it] = len(self.item2idx)
                self.idx2item[len(self.idx2item)] = it
            u_idx.append(self.user2idx[u])
            i_idx.append(self.item2idx[it])
            r_val.append(r)

        u = np.asarray(u_idx, dtype=np.int64)
        i = np.asarray(i_idx, dtype=np.int64)
        r = np.asarray(r_val, dtype=np.float32)

        return u, i, r

    @staticmethod
    def _split_per_user(
        u: np.ndarray,
        i: np.ndarray,
        r: np.ndarray,
        num_users: int,
        cfg: SplitConfig,
    ):
        if abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

        rng = np.random.default_rng(cfg.seed)

        user_rows: List[List[int]] = [[] for _ in range(num_users)]
        for idx, uu in enumerate(u):
            user_rows[int(uu)].append(idx)

        train_rows: List[int] = []
        val_rows: List[int] = []
        test_rows: List[int] = []

        for uu, rows in enumerate(user_rows):
            if not rows:
                continue
            rows = np.asarray(rows, dtype=np.int64)
            rng.shuffle(rows)

            if len(rows) < cfg.min_user_interactions:
                train_rows.extend(rows.tolist())
                continue

            n = len(rows)
            n_test = max(1, int(round(n * cfg.test_ratio)))
            n_val = max(1, int(round(n * cfg.val_ratio)))

            # ensure at least 1 train
            n_test = min(n_test, n - 2)
            n_val = min(n_val, n - 1 - n_test)

            test_part = rows[:n_test]
            val_part = rows[n_test:n_test + n_val]
            train_part = rows[n_test + n_val:]

            if len(train_part) == 0:
                # fallback
                train_part = rows[-1:]
                rest = rows[:-1]
                half = len(rest) // 2
                val_part = rest[:half]
                test_part = rest[half:]

            train_rows.extend(train_part.tolist())
            val_rows.extend(val_part.tolist())
            test_rows.extend(test_part.tolist())

        def pack(rows: List[int]) -> np.ndarray:
            if not rows:
                return np.empty((0, 3), dtype=np.float32)
            rows = np.asarray(rows, dtype=np.int64)
            out = np.stack([u[rows], i[rows], r[rows]], axis=1)
            return out.astype(np.float32)

        return pack(train_rows), pack(val_rows), pack(test_rows)

    def _save_meta(self, meta: ProcessedMeta) -> None:
        with open(os.path.join(self.processed_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
