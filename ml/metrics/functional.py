from typing import Tuple, Union

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

__all__ = ["mse", "mae", "r2_score"]

REDUCTIONS = {"mean", "none"}
DEFAULT_REDUCTION = "mean"


def raise_for_reduction(reduction: str):
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be in: {REDUCTIONS}, got: {reduction}")


@jaxtyped
@typechecker
def mse(
    output: Float[torch.Tensor, "batch out_features"],
    target: Float[torch.Tensor, "batch out_features"],
    reduction: str = DEFAULT_REDUCTION,
    return_grad: bool = False,
) -> Union[
    Float[torch.Tensor, "..."],
    Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "batch out_features"]],
]:
    raise_for_reduction(reduction)
    diff = output - target
    value = torch.square(diff)
    if reduction == "mean":
        value = value.mean()
    if not return_grad:
        return value
    d_output = 2 * diff / len(diff)
    d_output /= diff.shape[1]
    return value, d_output


@jaxtyped
@typechecker
def mae(
    output: Float[torch.Tensor, "batch out_features"],
    target: Float[torch.Tensor, "batch out_features"],
    reduction: str = DEFAULT_REDUCTION,
    return_grad: bool = False,
) -> Union[
    Float[torch.Tensor, "..."],
    Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "batch out_features"]],
]:
    raise_for_reduction(reduction)
    diff = output - target
    value = torch.abs(diff)
    if reduction == "mean":
        value = value.mean()
    if not return_grad:
        return value
    d_output = torch.sign(diff) / len(diff)
    d_output /= diff.shape[1]
    return value, d_output


def r2_score(
    output: Float[torch.Tensor, "batch out_features"],
    target: Float[torch.Tensor, "batch out_features"],
    reduction: str = DEFAULT_REDUCTION,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "out_channels"]]:
    raise_for_reduction(reduction)
    mean = target.mean(dim=0, keepdim=True)
    diff1 = (output - target).square().sum(dim=0)
    diff2 = (mean - target).square().sum(dim=0)
    value = diff1 / diff2
    if reduction == "mean":
        value = value.mean()
    return 1.0 - value
