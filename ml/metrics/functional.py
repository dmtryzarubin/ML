from typing import Callable, Tuple, Union

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped

from .. import activations

__all__ = ["mse", "mae", "r2_score", "get_stats"]

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


@jaxtyped
@typechecker
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


@jaxtyped
@typechecker
def binary_cross_entropy_with_logits(
    output: Float[torch.Tensor, "batch out_features"],
    target: Float[torch.Tensor, "batch out_features"],
    reduction: str = DEFAULT_REDUCTION,
    return_grad: bool = False,
) -> Union[
    Float[torch.Tensor, "..."],
    Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "batch out_features"]],
]:
    raise_for_reduction(reduction)
    probs = activations.sigmoid(output)
    value = target * torch.log(probs) + (1 - target) * torch.log(1 - probs)
    value *= -1.0
    if reduction == "mean":
        value = value.mean()
    if not return_grad:
        return value
    grad = probs - target
    grad /= len(output)
    grad /= output.shape[1]
    return value, grad


@jaxtyped
@typechecker
def get_stats(
    output: Int[torch.Tensor, "batch out_features"],
    target: Int[torch.Tensor, "batch out_features"],
) -> Tuple[
    Bool[torch.Tensor, "batch out_features"],
    Bool[torch.Tensor, "batch out_features"],
    Bool[torch.Tensor, "batch out_features"],
    Bool[torch.Tensor, "batch out_features"],
]:
    """
    Calculates TP, TN, FP, FN given binarized outputs

    :return: Tuple with TP, TN, FP, FN
    """
    tp = (output == target) & (target == 1)
    tn = (output == target) & (target == 0)
    fp = (output != target) & (target == 0)
    fn = (output != target) & (target == 1)
    return tp, tn, fp, fn


def _compute_metric(metric_fn: Callable, tp, tn, fp, fn, average: str, **metric_kwargs):
    if average == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, tn, fp, fn, **metric_kwargs)

    elif average == "macro":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, tn, fp, fn, **metric_kwargs)
        score = score.mean()

    return score


def _accuracy(
    tp, tn, fp, fn
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "out_channels"]]:
    return (tp + tn) / (tp + tn + fn + fp)


def accuracy(
    tp: torch.LongTensor,
    tn: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    average: str = "micro",
):
    return _compute_metric(_accuracy, tp, tn, fp, fn, average)
