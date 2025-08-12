import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std,a=-3 * std, b=3 * std,)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return x @ self.weight.t()
    

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_table = Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.embedding_table, 0.0, 1.0, -3.0, 3.0)

    def forward(
        self,
        x: torch.Tensor,
    ):
        return self.embedding_table[torch.clamp(x, 0, self.num_embeddings - 1)]
        