import torch
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.shape[0]

    start_idx = torch.randint(high=n - context_length, size=(batch_size,), dtype=torch.int64, device=device)
    start_idx = start_idx.unsqueeze(1)

    end_idx = torch.arange(context_length, dtype=torch.int32, device=device)
    end_idx = end_idx.unsqueeze(0)
    
    return (torch.Tensor(dataset[start_idx + end_idx]), torch.Tensor(dataset[(start_idx + 1) + end_idx]))
