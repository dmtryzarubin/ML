import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

__all__ = ["identity", "sigmoid", "relu"]


@jaxtyped
@typechecker
def identity(
    input: Float[torch.Tensor, "batch features"]
) -> Float[torch.Tensor, "batch features"]:
    return input


@jaxtyped
@typechecker
def sigmoid(
    input: Float[torch.Tensor, "batch features"]
) -> Float[torch.Tensor, "batch features"]:
    # 1 / (1 + e^-z)
    return (1 + torch.exp(-input)) ** -1.0


@jaxtyped
@typechecker
def relu(
    input: Float[torch.Tensor, "batch features"]
) -> Float[torch.Tensor, "batch features"]:
    return torch.maximum(input, torch.zeros_like(input))
