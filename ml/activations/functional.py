import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

__all__ = ["identity", "sigmoid", "softmax", "relu"]


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
def softmax(
    input: Float[torch.Tensor, "batch features"],
    dim: int,
) -> Float[torch.Tensor, "batch features"]:
    """
    Applies softmax function

    :input: Input tensor
    :return: Input tensor with sigmoid applied
    """
    maxes = torch.max(input, dim=dim, keepdim=True)[0]
    exp = torch.exp(input - maxes)
    exp_sum = torch.sum(exp, dim=dim, keepdim=True)
    output = exp / exp_sum
    return output


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
