from typing import Tuple

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

__all__ = ["linear", "_linear_forward", "_linear_backward"]


@jaxtyped
@typechecker
def linear(
    input: Float[torch.Tensor, "batch in_features"],
    weight: Float[torch.Tensor, "out_features in_features"],
    bias: Float[torch.Tensor, "out_features"],
) -> Float[torch.Tensor, "batch out_features"]:
    return input @ weight.T + bias


@jaxtyped
@typechecker
def _linear_forward(
    input: Float[torch.Tensor, "batch in_features"],
    weight: Float[torch.Tensor, "out_features in_features"],
    bias: Float[torch.Tensor, "out_features"],
) -> Tuple[
    Float[torch.Tensor, "batch out_features"],
    Tuple[
        Float[torch.Tensor, "batch in_features"],
        Float[torch.Tensor, "out_features in_features"],
        Float[torch.Tensor, "out_features"],
    ],
]:
    output = linear(input, weight, bias)
    return output, (input, weight, bias)


def _linear_backward(
    dout: Float[torch.Tensor, "batch out_features"],
    cache: Tuple[
        Float[torch.Tensor, "batch in_features"],
        Float[torch.Tensor, "out_features"],
    ],
):
    input, weight, bias = cache
    d_weight = dout.T @ input
    d_bias = dout.sum(dim=0)
    return d_weight, d_bias
