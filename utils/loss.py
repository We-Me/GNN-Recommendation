import torch
from typing import Optional

def bpr_total_loss_with_scores(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    pos_score_val: Optional[torch.Tensor] = None,
    neg_score_val: Optional[torch.Tensor] = None,
    reg_weight: float = 0.0,
    weight_mode: str = 'pos'  # 可选：'pos' 或 'diff'
) -> torch.Tensor:
    """
    通用 BPR 损失函数，支持正负样本评分权重。

    Args:
        user_emb: [batch_size, dim] 用户嵌入
        pos_emb: [batch_size, dim] 正样本嵌入
        neg_emb: [batch_size, dim] 负样本嵌入
        pos_score_val: [batch_size] 正样本评分（可为权重）
        neg_score_val: [batch_size] 负样本评分（可为权重）
        reg_weight: L2 正则化权重
        weight_mode: 权重模式：
            - 'pos'：用 pos_score 作为样本权重（推荐）
            - 'diff'：用 (pos_score - neg_score) 作为权重（需保证 >0）

    Returns:
        标量损失值
    """
    pos_score = torch.sum(user_emb * pos_emb, dim=1)
    neg_score = torch.sum(user_emb * neg_emb, dim=1)
    diff = pos_score - neg_score

    base_loss = -torch.log(torch.sigmoid(diff) + 1e-8)

    if pos_score_val is not None:
        if weight_mode == 'pos':
            weight = pos_score_val
        elif weight_mode == 'diff' and neg_score_val is not None:
            weight = pos_score_val - neg_score_val
        else:
            raise ValueError("Invalid weight_mode or missing neg_score_val.")
        weighted_loss = (weight * base_loss).mean()
    else:
        weighted_loss = base_loss.mean()

    reg_term = (
        user_emb.norm(p=2, dim=1).pow(2) +
        pos_emb.norm(p=2, dim=1).pow(2) +
        neg_emb.norm(p=2, dim=1).pow(2)
    ).mean()

    return weighted_loss + reg_weight * reg_term
