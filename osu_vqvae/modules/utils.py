from typing import Tuple

import torch
from torch.nn import functional as F


def pad_at_dim(
    t: torch.Tensor,
    pad: Tuple[int],
    dim: int = -1,
    value: float = 0.0,
) -> torch.Tensor:
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)
