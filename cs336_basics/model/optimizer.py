from collections.abc import Callable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        betas: tuple[float],
        eps: float,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps}
        super().__init__(params, defaults)

    def step(
        self,
        closure: Optional[Callable] = None,
    ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            betas = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data
                b1t = state.get('b1t', 1.0) * betas[0]
                b2t = state.get('b2t', 1.0) * betas[1]
                m = state.get('m', torch.zeros_like(p.data)) * betas[0] + (1 - betas[0]) * grad
                v = state.get('v', torch.zeros_like(p.data)) * betas[1] + (1 - betas[1]) * torch.pow(grad, 2)

                lr_at_t = lr * math.sqrt(1 - b2t) / (1 - b1t)

                p.data = p.data - lr_at_t * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * weight_decay * p.data

                state['b1t'] = b1t
                state['b2t'] = b2t
                state['m'] = m
                state['v'] = v
        return loss