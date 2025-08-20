import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.shape[0]

    start_idx = np.random.randint(low=0, high=n-context_length, size=(batch_size,), dtype=np.int64)
    start_idx = np.expand_dims(start_idx, axis=1)

    end_idx = np.arange(context_length, dtype=np.int32)
    end_idx = np.expand_dims(end_idx, axis=0)
    
    return (torch.tensor(dataset[start_idx + end_idx], dtype=torch.int32, device=device), torch.tensor(dataset[(start_idx + 1) + end_idx], dtype=torch.int32, device=device))
