from typing import Optional, Tuple

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped


def identity(
    param: Float[torch.Tensor, "..."], *args, **kwargs
) -> Float[torch.Tensor, "..."]:
    return param


@jaxtyped
@typechecker
def sgd(
    param: Float[torch.Tensor, "..."],
    grad: Float[torch.Tensor, "..."],
    lr: float,
    weight_decay: float = 0.0,
    *args,
    **kwargs,
) -> torch.FloatTensor:
    regularization = 2 * param.sum() * weight_decay
    param -= lr * (grad + regularization)
    return param


@jaxtyped
@typechecker
def adam(
    param: Float[torch.Tensor, "..."],
    grad: Float[torch.Tensor, "..."],
    m: Float[torch.Tensor, "..."],
    v: Float[torch.Tensor, "..."],
    lr: float,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    *args,
    **kwargs,
) -> Tuple[
    Float[torch.Tensor, "..."], Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]
]:
    beta1, beta2 = betas
    regularization = 2 * param.sum() * weight_decay
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    ma = m / (1 - beta1)
    va = v / (1 - beta2)

    param -= lr * (regularization + ma / torch.sqrt(va + eps))
    return param, m, v
