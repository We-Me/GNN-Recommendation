import torch
import pytorch_lightning as pl

from src.loss.BPRLoss import BPRLoss
from src.models.LightGCN import LightGCN
from src.metrics.metrics import Metric

class LightGCNModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        optim_cfg: dict,
        num_users: int,
        num_items: int,
        norm_adj: torch.Tensor,
        train_interactions: torch.Tensor,
        val_ground_truth: dict[int, set[int]],
        test_ground_truth: dict[int, set[int]],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["norm_adj", "train_interactions", "val_ground_truth", "test_ground_truth"])

        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.num_users = num_users
        self.num_items = num_items
        self.num_fold = model_cfg.get("num_fold", 100)
        self.train_interactions = train_interactions
        self.val_gt = val_ground_truth
        self.test_gt = test_ground_truth
        self.topk = list(model_cfg.get("topk", [10, 20]))

        self.model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embed_dim=model_cfg["embed_dim"],
            num_layers=model_cfg["num_layers"],
            norm_adj=norm_adj,
            num_fold=model_cfg["num_fold"],
            dropout_flag=model_cfg["dropout_flag"],
            dropout_rate=model_cfg["dropout_rate"]
        )

        self.loss = BPRLoss(model_cfg.get("reg_ratio", 1.0))

    def forward(self):
        return self.model()
    
    def configure_optimizers(self):
        name = self.optim_cfg.get("optimizer", "adam").lower()
        lr = self.optim_cfg.get("lr", 0.001)
        weight_decay = self.optim_cfg.get("weight_decay", 0)

        if name == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

        return opt

    def training_step(self, batch, batch_idx):
        u, pos_i, neg_i = batch
        user_emb, item_emb = self()

        u_emb = user_emb[u]
        pos_emb = item_emb[pos_i]
        neg_emb = item_emb[neg_i]

        loss = self.loss.get(u_emb, pos_emb, neg_emb)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=u.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        return None
    
    def on_validation_epoch_end(self):
        self._evaluate(split="val")

    def test_step(self, batch, batch_idx):
        return None

    def on_test_epoch_end(self):
        self._evaluate(split="test")

    @torch.no_grad()
    def _evaluate(self, split: str) -> None:
        gt = self.val_gt if split == "val" else self.test_gt
        if not gt:
            return

        user_emb, item_emb = self()
        scores = user_emb @ item_emb.T  # [U, I]
        scores = self._mask_train_seen(scores, self.train_interactions)

        max_k = max(self.topk)
        _, top_idx = torch.topk(scores, k=max_k, dim=1)
        top_idx = top_idx.cpu().tolist()
        rec = {u: top_idx[u] for u in range(len(top_idx))}

        num_users = user_emb.size(0)
        for k in self.topk:
            hits = Metric.hits(gt, rec, k)
            r = Metric.recall(gt, hits)
            p = Metric.precision(hits, num_users, k)
            n = Metric.ndcg(gt, rec, k)

            self.log(
                f"{split}/Recall@{k}",
                r,
                on_step=False,
                on_epoch=True,
                prog_bar=(split == "val" and k == 20),
                sync_dist=True,
            )
            self.log(
                f"{split}/Precision@{k}",
                p,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{split}/NDCG@{k}",
                n,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    @staticmethod
    def _mask_train_seen(scores: torch.Tensor, train_interactions: torch.Tensor) -> torch.Tensor:
        train_interactions = train_interactions.coalesce()
        u = train_interactions.indices()[0]
        i = train_interactions.indices()[1]
        scores[u, i] = float("-inf")
        return scores
