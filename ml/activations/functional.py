import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

__all__ = ["identity", "sigmoid", "relu"]


@jaxtyped
@typechecker
def identity(
    input: Float[torch.Tensor, "batch features"]
) -> Float[torch.Tensor, "batch features"]:
    """
    Identity function

    :input: Input tensor
    :return: Input tensor
    """
    return input


@jaxtyped
@typechecker
def sigmoid(
    input: Float[torch.Tensor, "batch features"]
) -> Float[torch.Tensor, "batch features"]:
    """
    Applies sigmoid function to each element
    output = 1 / (1 + e^{-input})

    :input: Input tensor
    :return: Input tensor with sigmoid applied
    """
    return (1 + torch.exp(-input)) ** -1.0


@jaxtyped
@typechecker
def relu(
    input: Float[torch.Tensor, "batch features"]
) -> Float[torch.Tensor, "batch features"]:
    """
    Thresholds input at zero
    output = max(input, 0)

    :input: Input tensor
    :return: Input tensor with ReLU applied
    """
    return torch.maximum(input, torch.zeros_like(input))
