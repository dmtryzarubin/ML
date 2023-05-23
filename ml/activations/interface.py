import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

from .functional import *

__all__ = ["Activation", "Sigmoid", "ReLU"]


class Activation:
    """
    Base container for forward and backward pass of some activation
    """

    _parameters = []
    _grad = []

    def parameters(self):
        return self._parameters

    def grad(self):
        return self._grad

    def forward(self, input):
        return input

    def backward(self, dout):
        return dout


class Sigmoid(Activation):
    """
    Class for Sigmoid forward and backward pass
    """

    _name = "sigmoid"

    def forward(self, input: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        self.cache = sigmoid(input)
        return self.cache

    def backward(self, dout: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return dout * self.cache * (1 - self.cache)


class ReLU(Activation):
    """
    Class for ReLU forward and backward pass
    """

    _name = "relu"

    def forward(self, input: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        output = relu(input)
        self.cache = (output > 0).to(input)
        return output

    def backward(self, dout: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return dout * self.cache
