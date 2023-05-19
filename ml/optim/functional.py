import torch


def identity(param, *args, **kwargs):
    return param


def sgd(
    param: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    weight_decay: float = 0.0,
    *args,
    **kwargs
) -> torch.Tensor:
    regularization = 2 * param.sum() * weight_decay
    param -= lr * (grad + regularization)
    return param
