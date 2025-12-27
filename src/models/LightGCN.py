import torch
import torch.nn as nn

from .utils import slice_sparse_rows, sparse_dropout


class LightGCN(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embed_dim: int,
                 num_layers: int,
                 norm_adj: torch.Tensor,
                 num_fold: int = 1,
                 dropout_flag: bool = False,
                 dropout_rate: float = 0.0):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_fold = num_fold
        self.norm_adj = norm_adj
        self.dropout_flag = dropout_flag
        self.dropout_rate = dropout_rate

        self.embedding_dict = nn.ParameterDict({
            "user_embedding": nn.Parameter(torch.empty(num_users, embed_dim)),
            "item_embedding": nn.Parameter(torch.empty(num_items, embed_dim))
        })
        nn.init.xavier_uniform_(self.embedding_dict["user_embedding"])
        nn.init.xavier_uniform_(self.embedding_dict["item_embedding"])
    
    def load_pretrain_emdeddings(self, pretrain_user_embedding: torch.Tensor, pretrain_item_embedding: torch.Tensor):
        assert pretrain_user_embedding.shape == self.embedding_dict["user_embedding"].shape
        assert pretrain_item_embedding.shape == self.embedding_dict["item_embedding"].shape
        self.embedding_dict["user_embedding"].data.copy_(pretrain_user_embedding)
        self.embedding_dict["item_embedding"].data.copy_(pretrain_item_embedding)

    def _split_A_hat_node_dropout(self, X: torch.Tensor):
        A_fold_hat = []
        dev = X.device
        num_nodes = self.num_users + self.num_items
        fold_len = num_nodes // self.num_fold

        for i_fold in range(self.num_fold):
            start = i_fold * fold_len
            end = num_nodes if i_fold == self.num_fold - 1 else (i_fold + 1) * fold_len

            temp = slice_sparse_rows(X, start, end)
            if temp.device != dev:
                temp = temp.to(dev)

            if self.dropout_flag and self.training:
                temp = sparse_dropout(temp, self.dropout_rate)
                if temp.device != dev:
                    temp = temp.to(dev)

            A_fold_hat.append(temp)

        return A_fold_hat

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict["user_embedding"], self.embedding_dict["item_embedding"]], dim=0)
        dev = ego_embeddings.device

        A = self.norm_adj
        if A.device != dev:
            A = A.to(dev)
        A_fold_hat = self._split_A_hat_node_dropout(A)

        all_embeddings = [ego_embeddings]

        for _ in range(self.num_layers):
            temp_emb = []
            for f in range(len(A_fold_hat)):
                temp_emb.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))
            ego_embeddings = torch.cat(temp_emb, dim=0)
            all_embeddings.append(ego_embeddings)

        stacked_embeddings = torch.stack(all_embeddings, dim=1)
        out_embedding = torch.mean(stacked_embeddings, dim=1, keepdim=False)
        user_g_embedding, item_g_embedding = torch.split(out_embedding, [self.num_users, self.num_items], dim=0)

        return user_g_embedding, item_g_embedding
