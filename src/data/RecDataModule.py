import os, json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import scipy as sp
import torch
from torch.utils.data import DataLoader

from .BPRDataset import BPRDataset


@dataclass
class DataConfig:
    processed_dir: str
    batch_size: int
    num_workers: int
    num_negatives: int

class RecDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig, seed: int = 42):
        super().__init__()
        self.cfg = cfg
        self.seed = seed

        self.num_users: Optional[int] = None
        self.num_items: Optional[int] = None
        self.train_ds: Optional[BPRDataset] = None
        self.norm_adj: Optional[torch.Tensor] = None
        self.train_interaction_matrix: Optional[torch.Tensor] = None
        self.val_gt: Optional[dict] = None
        self.test_gt: Optional[dict] = None

    def setup(self, stage: Optional[str] = None):
        meta_path = os.path.join(self.cfg.processed_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.num_users = int(meta["num_users"])
        self.num_items = int(meta["num_items"])

        # train.npy: float32 [N,3] (u,i,rating) - rating ignored by BPRDataset
        train = np.load(os.path.join(self.cfg.processed_dir, "train.npy"))
        self.train_ds = BPRDataset(
            train_data=train,
            num_items=self.num_items,
            num_negatives=self.cfg.num_negatives,
            seed=self.seed,
        )
        self.norm_adj = self._build_normalized_adj_matrix(train, self.num_users, self.num_items)
        self.train_interaction_matrix = self._build_interaction_matrix(train, self.num_users, self.num_items)

        self.val_gt = self._build_gt(os.path.join(self.cfg.processed_dir, "val.npy"))
        self.test_gt = self._build_gt(os.path.join(self.cfg.processed_dir, "test.npy"))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
        )

    def get_norm_adj(self):
        return self.norm_adj
    
    def get_train_interactions(self):
        return self.train_interaction_matrix
    
    def get_gt(self):
        return self.val_gt, self.test_gt
    
    @staticmethod
    def _build_gt(path: str) -> dict[int, set[int]]:
        arr = np.load(path)
        gt: dict[int, set[int]] = {}
        if arr.size == 0:
            return gt
        u = arr[:, 0].astype(np.int64)
        i = arr[:, 1].astype(np.int64)
        for uu, ii in zip(u, i):
            gt.setdefault(int(uu), set()).add(int(ii))
        return gt

    @staticmethod
    def _build_normalized_adj_matrix(data: np.ndarray, num_users: int, num_items: int) -> torch.Tensor:
        assert data.ndim == 2 and data.shape[1] in [2, 3]

        user_ids = data[:, 0].astype(np.int64)
        item_ids = data[:, 1].astype(np.int64) + num_users
        if data.shape[1] == 3:
            weights = data[:, 2].astype(np.float32)
        else:
            weights = np.ones(len(user_ids), dtype=np.float32)

        num_nodes = num_users + num_items

        # 对称添加边 (u, i) 与 (i, u)
        row = np.concatenate([user_ids, item_ids])
        col = np.concatenate([item_ids, user_ids])
        data_val = np.concatenate([weights, weights])

        adj = sp.sparse.coo_matrix((data_val, (row, col)), shape=(num_nodes, num_nodes))

        # D^{-1/2} A D^{-1/2}
        deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
        deg[deg == 0] = 1.0
        deg_inv_sqrt = np.power(deg, -0.5)
        d_inv_sqrt = sp.sparse.diags(deg_inv_sqrt)
        norm_adj = d_inv_sqrt.dot(adj).dot(d_inv_sqrt).tocoo()

        indices = torch.from_numpy(np.vstack([norm_adj.row, norm_adj.col]).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))

        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    @staticmethod
    def _build_interaction_matrix(data: np.ndarray, num_users: int, num_items: int) -> torch.Tensor:
        assert data.ndim == 2 and data.shape[1] in (2, 3)

        user_ids = data[:, 0].astype(np.int64)
        item_ids = data[:, 1].astype(np.int64)
        values = np.ones(len(user_ids), dtype=np.float32)

        r = sp.sparse.coo_matrix((values, (user_ids, item_ids)), shape=(num_users, num_items))

        indices = torch.from_numpy(np.vstack([r.row, r.col]).astype(np.int64))
        vals = torch.from_numpy(r.data.astype(np.float32))

        return torch.sparse_coo_tensor(indices, vals, (num_users, num_items)).coalesce()
