import torch
import torch.nn as nn

from utils.matrix import sparse_dropout


class LightGCN(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embed_dim: int,
                 num_layers: int,
                 norm_adj: torch.Tensor,
                 dropout: float = 0.0,
                 pretrain_user_emb: torch.Tensor = None,
                 pretrain_item_emb: torch.Tensor = None):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.register_buffer('norm_adj', norm_adj)

        self.embedding_dict = nn.ParameterDict({
            "user_emb": nn.Parameter(torch.empty(num_users, embed_dim)),
            "item_emb": nn.Parameter(torch.empty(num_items, embed_dim))
        })

        if pretrain_user_emb is not None and pretrain_item_emb is not None:
            assert pretrain_user_emb.shape == self.embedding_dict["user_emb"].shape
            assert pretrain_item_emb.shape == self.embedding_dict["item_emb"].shape
            self.embedding_dict["user_emb"].data.copy_(pretrain_user_emb)
            self.embedding_dict["item_emb"].data.copy_(pretrain_item_emb)
        else:
            nn.init.xavier_uniform_(self.embedding_dict["user_emb"])
            nn.init.xavier_uniform_(self.embedding_dict["item_emb"])

    def forward(self, return_all_layers=False):
        if self.training and self.dropout > 0:
            norm_adj = sparse_dropout(self.norm_adj, self.dropout)
        else:
            norm_adj = self.norm_adj

        ego_embeddings = torch.cat([self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.num_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        stacked_embeddings = torch.stack(all_embeddings, dim=1)
        out_embedding = torch.mean(stacked_embeddings, dim=1)

        user_final = out_embedding[:self.num_users]
        item_final = out_embedding[self.num_users:]
        if return_all_layers:
            return user_final, item_final, stacked_embeddings
        return user_final, item_final
