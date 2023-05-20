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
    """
    Performs affine transformation Wx + b

    :return: tensor with affine applied
    """
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
    ],
]:
    """
    Performs affine transformation Wx + b
    and stores the cache to use for gradient calculation

    :return: Tuple with `output` and tuple with cached variables (`input`, `weight`, `bias`)
    """
    output = linear(input, weight, bias)
    return output, (input, weight)


def _linear_backward(
    dout: Float[torch.Tensor, "batch out_features"],
    cache: Tuple[
        Float[torch.Tensor, "batch in_features"],
        Float[torch.Tensor, "batch in_features"],
    ],
):
    """
    Calculates gradient w.r.t to `input`, `weight`, `bias`

    :param dout: Outer gradient that is then multiplied by local gradient
    :param cache: Tuple of cached variables `input`, `weight`, `bias`
    :return: Tuple with gradients for `input`, `weight`, `bias`
    """
    input, weight = cache
    d_input = dout @ weight
    d_weight = dout.T @ input
    d_bias = dout.sum(dim=0)
    return d_input, d_weight, d_bias
