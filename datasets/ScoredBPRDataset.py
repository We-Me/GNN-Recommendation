import random
from typing import Dict, Set

import torch
from torch.utils.data import Dataset


class ScoredBPRDataset(Dataset):
    def __init__(self,
                 interaction_matrix: torch.Tensor,
                 num_negatives: int = 1,
                 threshold: float = 0.0):
        super().__init__()
        assert interaction_matrix.is_sparse, "必须为稀疏矩阵"

        self.interaction_matrix = interaction_matrix.coalesce()
        self.num_users, self.num_items = interaction_matrix.shape
        self.num_negatives = num_negatives
        self.threshold = threshold

        # 构建正负样本索引和得分映射
        self.user_pos_dict, self.user_pos_score = self._build_user_dict(greater=True)
        self.user_neg_dict, self.user_neg_score = self._build_user_dict(greater=False)

        # 缓存负样本列表用于快速采样
        self.user_neg_list = {
            u: list(items) for u, items in self.user_neg_dict.items() if items
        }

        # 补集备用
        self.user_all_items = {
            u: set(range(self.num_items)) - self.user_pos_dict.get(u, set())
            for u in range(self.num_users)
        }

        # 正样本对缓存
        self.user_pos_pairs = self._build_user_pos_pairs()
        self.total_triplets = len(self.user_pos_pairs) * num_negatives

    def _build_user_dict(self, greater: bool):
        user_dict: Dict[int, Set[int]] = {}
        user_score: Dict[int, Dict[int, float]] = {}

        indices = self.interaction_matrix.indices()
        values = self.interaction_matrix.values()

        for idx in range(values.size(0)):
            u = indices[0, idx].item()
            i = indices[1, idx].item()
            score = values[idx].item()
            if (greater and score > self.threshold) or (not greater and score <= self.threshold):
                user_dict.setdefault(u, set()).add(i)
                user_score.setdefault(u, {})[i] = score

        return user_dict, user_score

    def _build_user_pos_pairs(self):
        return [(u, i) for u, items in self.user_pos_dict.items() for i in items]

    def _sample_neg(self, user: int):
        if user in self.user_neg_list:
            neg_item = random.choice(self.user_neg_list[user])
            neg_score = self.user_neg_score[user][neg_item]
        else:
            # 从全集中随机选负样本（极少触发）
            candidates = self.user_all_items.get(user, set())
            while True:
                neg_item = random.randint(0, self.num_items - 1)
                if neg_item in candidates:
                    neg_score = 0.0
                    break
        return neg_item, neg_score

    def __len__(self):
        return self.total_triplets

    def __getitem__(self, idx: int):
        base_idx = idx // self.num_negatives
        user, pos_item = self.user_pos_pairs[base_idx]
        pos_score = self.user_pos_score[user][pos_item]
        neg_item, neg_score = self._sample_neg(user)
        return user, pos_item, pos_score, neg_item, neg_score
