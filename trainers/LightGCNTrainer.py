import os
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.RawProcessor import RawProcessor
from datasets.ScoredBPRDataset import ScoredBPRDataset
from trainers.Trainer import BaseTrainer
from utils.loss import bpr_total_loss_with_scores
from utils.metrics import ranking_evaluation
from utils.utils import import_class_from_string


class LightGCNTrainer(BaseTrainer):
    def __init__(self, options: dict[str, Any]):
        super(LightGCNTrainer, self).__init__(options)
        train_cfg = options['train_cfg']
        self.device = train_cfg['device']
        self.epochs = train_cfg['epochs']
        self.batch_size = train_cfg['batch_size']
        self.reg_radio = train_cfg['reg_radio']
        self.eval_freq = train_cfg['eval_freq']
        self.topk = train_cfg['topk']
        self.save_dir = train_cfg['save_dir']
        self.use_scheduler = 'scheduler_cfg' in options

        dataset_cfg = options['dataset_cfg']
        rp = RawProcessor(**dataset_cfg)
        self.data = rp.get_data()
        self.train_data, self.test_data = rp.get_train_test()
        self.num_users = rp.get_num_users()
        self.num_items = rp.get_num_items()
        self.idx2user = rp.get_idx2user()
        self.idx2item = rp.get_idx2item()
        self.interaction_matrix = rp.get_interaction_matrix()
        self.norm_adj = rp.get_norm_adj()

        model_cfg = options['model_cfg']
        model_class, class_name = import_class_from_string(model_cfg['class'])
        assert class_name == 'LightGCN'
        self.model = model_class(num_users=self.num_users,
                                 num_items=self.num_items,
                                 norm_adj=self.norm_adj,
                                 **model_cfg['args'])

        optimizer_cfg = options['optimizer_cfg']
        optimizer_class, _ = import_class_from_string(optimizer_cfg['class'])
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_cfg['args'])

        if self.use_scheduler:
            scheduler_cfg = options['scheduler_cfg']
            scheduler_class, _ = import_class_from_string(scheduler_cfg['class'])
            self.scheduler = scheduler_class(self.optimizer, **scheduler_cfg['args'])
        else:
            self.scheduler = None

        self.best_user_emb = None
        self.best_item_emb = None

    def train(self):
        train_dataset = ScoredBPRDataset(self.interaction_matrix)
        train_loader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle=True,
            drop_last=True
        )

        self.model.to(self.device)
        best_recall = float('-inf')
        for epoch in range(self.epochs):
            self.model.train()
            train_loader_tqdm = tqdm(train_loader, desc=f"Training - Epoch {epoch}/{self.epochs}")
            for user, pos_item, pos_score, neg_item, neg_score in train_loader_tqdm:
                user = user.to(self.device)
                pos_item = pos_item.to(self.device)
                pos_score = pos_score.to(self.device)
                neg_item = neg_item.to(self.device)
                neg_score = neg_score.to(self.device)

                user_emb, item_emb = self.model()

                usr = user_emb[user]
                pos = item_emb[pos_item]
                neg = item_emb[neg_item]

                loss = bpr_total_loss_with_scores(
                    user_emb=usr,
                    pos_emb=pos,
                    neg_emb=neg,
                    pos_score_val=pos_score,
                    neg_score_val=neg_score,
                    reg_weight=self.reg_radio,
                    weight_mode='diff'
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loader_tqdm.set_postfix(loss=loss.detach().item(), best_recall=best_recall)

            if self.use_scheduler:
                self.scheduler.step()

            if (epoch + 1) % self.eval_freq == 0:
                with torch.no_grad():
                    user_emb, item_emb = self.model()

                report, topk = self.eval(user_emb, item_emb)

                if report[topk[0]]['Recall'] > best_recall:
                    best_recall = report[topk[0]]['Recall']
                    self.best_user_emb = user_emb.clone()
                    self.best_item_emb = item_emb.clone()
                    self.save_model()

    def eval(self, user_emb, item_emb):
        self.model.eval()

        if isinstance(self.topk, list):
            topk = self.topk
        elif isinstance(self.topk, int):
            topk = [self.topk]
        else:
            raise ValueError

        ground_truth = {}
        for row in self.test_data:
            user = int(row[0])
            item = int(row[1])
            ground_truth.setdefault(user, {})[item] = 1

        scores = torch.matmul(user_emb, item_emb.T)
        train_matrix = self.interaction_matrix.to(scores.device)
        if train_matrix.is_sparse:
            train_mask = train_matrix.to_dense() > 0
        else:
            train_mask = train_matrix > 0
        scores = scores.masked_fill(train_mask, -1e10)

        # 获取 top-K 推荐结果
        _, top_items = torch.topk(scores, k=max(topk), dim=-1)
        top_items = top_items.cpu().numpy()

        # 构建推荐结果，仅对 ground_truth 中的用户
        result = {
            u: [(int(i), 1.0) for i in top_items[u]]
            for u in ground_truth
        }

        report = ranking_evaluation(ground_truth, result, topk)

        for n in topk:
            print(f"Top {n}")
            print(f"Hit Ratio:  {report[n]['Hit Ratio']:.4f}")
            print(f"Precision:  {report[n]['Precision']:.4f}")
            print(f"Recall:     {report[n]['Recall']:.4f}")
            print(f"NDCG:       {report[n]['NDCG']:.4f}")

        return report, topk

    def save_model(self):
        if self.model is None:
            raise RuntimeError("模型尚未初始化或训练，无法保存。")
        os.makedirs(self.save_dir, exist_ok=True)

        save_dict = {
            "model_state": self.model.state_dict(),
            "best_user_emb": self.best_user_emb.cpu() if self.best_user_emb is not None else None,
            "best_item_emb": self.best_item_emb.cpu() if self.best_item_emb is not None else None,
            "option": self.option['model_cfg'],
            "idx2user": self.idx2user,
            "idx2item": self.idx2item
        }
        save_file = os.path.join(self.save_dir, f"ckpt.pt")
        torch.save(save_dict, save_file)
