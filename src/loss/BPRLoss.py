import torch


class BPRLoss(object):
    def __init__(self, reg_ratio: float):
        self.reg_ratio = reg_ratio

    @staticmethod
    def bpr_loss(u_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

    @staticmethod
    def l2_reg(*tensors: torch.Tensor) -> torch.Tensor:
        reg = torch.zeros((), device=tensors[0].device)
        for t in tensors:
            reg = reg + (t.pow(2).sum(dim=-1).mean())
        return reg
    
    def get(self, u_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        return BPRLoss.bpr_loss(u_emb, pos_emb, neg_emb) + \
            self.reg_ratio * BPRLoss.l2_reg(u_emb, pos_emb, neg_emb)
    