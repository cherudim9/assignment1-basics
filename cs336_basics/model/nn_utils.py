import torch
from torch import Tensor
from collections.abc import Iterable
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


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    norm2 = 0.0
    for param in parameters:
        if param.requires_grad:
            norm2 += torch.sum(torch.square(param.grad))
    norm2 = norm2 ** 0.5
    if norm2 > max_l2_norm:
        coef = max_l2_norm / (norm2 + 1e-6)
        for param in parameters:
            if param.requires_grad:
                param.grad = param.grad * coef