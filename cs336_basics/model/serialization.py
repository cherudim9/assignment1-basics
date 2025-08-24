import os
from typing import IO, BinaryIO
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    ret = {}
    ret['iteration'] = iteration
    ret['model'] = model.state_dict()
    ret['optimizer'] = optimizer.state_dict()
    torch.save(ret, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> int:
    d = torch.load(src, map_location=torch.device('cpu'))
    model.load_state_dict(d['model'])
    if optimizer is not None:
        optimizer.load_state_dict(d['optimizer'])
    return d['iteration']
