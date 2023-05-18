import torch


def identity(param, *args, **kwargs):
    return param


def sgd(
    param: torch.Tensor, grad: torch.Tensor, lr: float, *args, **kwargs
) -> torch.Tensor:
    param -= grad * lr
    return param
