import numpy as np
import scipy as sp
import torch


def build_interaction_matrix(interaction_data: torch.Tensor, num_users: int, num_items: int) -> torch.Tensor:
    assert interaction_data.ndim == 2 and interaction_data.shape[1] == 2, "输入必须是 [N, 2] 的张量"

    user_ids = interaction_data[:, 0].long()
    item_ids = interaction_data[:, 1].long()

    indices = torch.stack([user_ids, item_ids], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    interaction_matrix = torch.sparse_coo_tensor(indices, values, size=(num_users, num_items))
    return interaction_matrix.coalesce()


def build_weighted_interaction_matrix(interaction_data: torch.Tensor, num_users: int, num_items: int) -> torch.Tensor:
    assert interaction_data.ndim == 2 and interaction_data.shape[1] == 3, "输入必须是 [N, 3] 的张量"

    user_ids = interaction_data[:, 0].long()
    item_ids = interaction_data[:, 1].long()
    weights = interaction_data[:, 2].float()

    indices = torch.stack([user_ids, item_ids], dim=0)

    interaction_matrix = torch.sparse_coo_tensor(indices, weights, size=(num_users, num_items))
    return interaction_matrix.coalesce()

def build_normalized_adj_matrix(data: torch.Tensor, num_users: int, num_items: int) -> torch.Tensor:
    assert data.ndim == 2 and data.shape[1] in [2, 3], "输入必须是形如 [N, 2] 或 [N, 3] 的张量"

    user_ids = data[:, 0].long().cpu().numpy()
    item_ids = data[:, 1].long().cpu().numpy() + num_users  # 偏移 item idx
    if data.shape[1] == 3:
        weights = data[:, 2].float().cpu().numpy()
    else:
        weights = np.ones_like(user_ids, dtype=np.float32)

    # 对称添加边 (u, i) 与 (i, u)
    row = np.concatenate([user_ids, item_ids])
    col = np.concatenate([item_ids, user_ids])
    data_val = np.concatenate([weights, weights])

    num_nodes = num_users + num_items
    adj = sp.sparse.coo_matrix((data_val, (row, col)), shape=(num_nodes, num_nodes))

    # 计算 D^{-1/2}
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0  # 防止除以 0
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0  # 安全处理

    d_inv_sqrt = sp.sparse.diags(deg_inv_sqrt)

    norm_adj = d_inv_sqrt.dot(adj).dot(d_inv_sqrt).tocoo()

    indices = torch.from_numpy(
        np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
    )
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = norm_adj.shape

    return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()


def sparse_dropout(sparse_mat: torch.Tensor, dropout_rate: float):
    nnz = sparse_mat._nnz()
    noise = torch.rand(nnz, device=sparse_mat.device)
    keep_mask = noise >= dropout_rate
    indices = sparse_mat._indices()[:, keep_mask]
    values = sparse_mat._values()[keep_mask] * (1.0 / (1 - dropout_rate))
    return torch.sparse_coo_tensor(indices, values, sparse_mat.shape, device=sparse_mat.device).coalesce()
