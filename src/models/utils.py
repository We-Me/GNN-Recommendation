import torch


def slice_sparse_rows(sparse_mat: torch.Tensor, start: int, end: int) -> torch.Tensor:
    assert sparse_mat.is_sparse, "matrix must be sparse matrix"
    sparse_mat = sparse_mat.coalesce()

    idx = sparse_mat.indices()   # [2, nnz]
    val = sparse_mat.values()    # [nnz]
    rows = idx[0]
    cols = idx[1]

    mask = (rows >= start) & (rows < end)
    new_rows = rows[mask] - start
    new_cols = cols[mask]
    new_idx = torch.stack([new_rows, new_cols], dim=0)
    new_val = val[mask]

    return torch.sparse_coo_tensor(
        new_idx, new_val,
        size=(end - start, sparse_mat.size(1)),
        device=sparse_mat.device,
        dtype=sparse_mat.dtype
    ).coalesce()

def sparse_dropout(sparse_mat: torch.Tensor, dropout_rate: float):
    assert sparse_mat.is_sparse, "matrix must be sparse matrix"
    assert 0.0 <= dropout_rate < 1.0, "dropout rate must within [0.0, 1.0)"
    sparse_mat = sparse_mat.coalesce()

    nnz = sparse_mat._nnz()
    if nnz == 0 or dropout_rate == 0.0:
        return sparse_mat

    noise = torch.rand(nnz, device=sparse_mat.device)
    keep_mask = noise >= dropout_rate

    indices = sparse_mat._indices()[:, keep_mask]
    values = sparse_mat._values()[keep_mask] * (1.0 / (1 - dropout_rate))

    return torch.sparse_coo_tensor(
        indices, values, sparse_mat.shape,
        device=sparse_mat.device
    ).coalesce()
