import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from math import sin, cos


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
    ) -> torch.Tensor:
        return self.embedding_table[torch.clamp(x, 0, self.num_embeddings - 1)]


class RmsNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = Parameter(torch.empty(d_model, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.gain, 1.0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.sum(x * x, dim = -1, keepdim = True) / self.d_model + self.eps)

        result = x / rms * self.gain

        return result.to(in_dtype)


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


class Swiglu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Parameter(torch.empty(d_ff, d_model, **factory_kwargs))
        self.w2 = Parameter(torch.empty(d_model, d_ff, **factory_kwargs))
        self.w3 = Parameter(torch.empty(d_ff, d_model, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w1, 0.0, 1.0, -3.0, 3.0)
        nn.init.trunc_normal_(self.w2, 0.0, 1.0, -3.0, 3.0)
        nn.init.trunc_normal_(self.w3, 0.0, 1.0, -3.0, 3.0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return (silu(x @ self.w1.t()) * (x @ self.w3.t())) @ self.w2.t()
    

class Rope(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        factory_kwargs = {"device": device}
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        assert self.d_k % 2 == 0
        self.reset_parameters()

    def reset_parameters(self):
        r_even = []
        r_odd = []
        for i in range(self.max_seq_len):
            a = []
            b = []
            for k in range(self.d_k // 2):
                theta_at_i_k = i / self.theta**(2.0 * k / self.d_k)
                a += [cos(theta_at_i_k), -sin(theta_at_i_k)]
                b += [sin(theta_at_i_k), cos(theta_at_i_k)]
            r_even.append(a)
            r_odd.append(b)
        self.register_buffer('r_even', torch.tensor(r_even), persistent=False)
        self.register_buffer('r_odd', torch.tensor(r_odd), persistent=False)

    def forward(
        self,
        x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:
        token_positions = token_positions - torch.min(token_positions, dim = -1).values
        r_even = self.r_even[token_positions]
        r_odd = self.r_odd[token_positions]

        x0 = x * r_even
        x0 = x0[..., ::2] + x0[..., 1::2]

        x1 = x * r_odd
        x1 = x1[..., ::2] + x1[..., 1::2]

        return torch.stack([x0, x1], dim=-1).flatten(start_dim=-2)
