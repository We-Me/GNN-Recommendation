import random
from typing import Dict, Set, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    def __init__(
        self,
        train_data: Union[np.ndarray, torch.Tensor],
        num_items: int,
        num_negatives: int = 1,
        seed: int = 42,
        max_tries: int = 50,
    ):
        super().__init__()
        if isinstance(train_data, torch.Tensor):
            data = train_data.detach().cpu().numpy()
        else:
            data = np.asarray(train_data)

        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("train_data must be 2D and have at least 2 columns: (user_idx, item_idx)")

        self.pos_users = data[:, 0].astype(np.int64)
        self.pos_items = data[:, 1].astype(np.int64)

        self.num_items = int(num_items)
        self.num_negatives = int(num_negatives)
        self.max_tries = int(max_tries)

        self._rng = random.Random(seed)

        # train 内的正集合（排除用）
        self.user_pos: Dict[int, Set[int]] = {}
        for u, i in zip(self.pos_users, self.pos_items):
            u = int(u)
            i = int(i)
            self.user_pos.setdefault(u, set()).add(i)

        self.total = len(self.pos_users) * self.num_negatives

    def __len__(self):
        return self.total

    def _sample_neg(self, u: int) -> int:
        seen = self.user_pos.get(u, set())

        for _ in range(self.max_tries):
            j = self._rng.randrange(self.num_items)
            if j not in seen:
                return j

        # 兜底：线性扫描（极少触发）
        for j in range(self.num_items):
            if j not in seen:
                return j

        # degenerate：看过所有 item
        return self._rng.randrange(self.num_items)

    def __getitem__(self, idx: int):
        base_idx = idx // self.num_negatives
        u = int(self.pos_users[base_idx])
        i = int(self.pos_items[base_idx])
        j = self._sample_neg(u)

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long),
        )
