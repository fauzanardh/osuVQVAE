from functools import wraps
from typing import Tuple

import torch
from einops import rearrange
from torch.nn import functional as F


def once(fn: callable) -> callable:
    called = False

    @wraps(fn)
    def inner(x: str) -> None:
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def log(t: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(t.dtype).eps
    return torch.log(t + eps)


def gradient_penalty(
    sig: torch.Tensor,
    output: torch.Tensor,
    weight: int = 10,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=sig,
        grad_outputs=torch.ones_like(output, device=sig.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = rearrange(gradients, "b ... -> b (...)")
    return weight * ((gradients.norm(2, dim=-1) - 1) ** 2).mean()


def pad_at_dim(
    t: torch.Tensor,
    pad: Tuple[int],
    dim: int = -1,
    value: float = 0.0,
) -> torch.Tensor:
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)
