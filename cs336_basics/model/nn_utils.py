import torch
from torch import Tensor
from jaxtyping import Float, Int


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x


def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
):
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    s = -inputs[torch.arange(inputs.shape[0]), targets] + torch.log(torch.sum(torch.exp(inputs), dim=-1))
    return torch.mean(s)