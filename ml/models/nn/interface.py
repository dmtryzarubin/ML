from typing import Callable, List, Union

import torch
from beartype import beartype as typechecker
from jaxtyping import Float
from tqdm.notebook import tqdm

from ...activations import Activation
from ...losses import Loss
from ...optim import Optimizer
from ..linear import Linear
from ..linear.functional import _linear_backward


class Sequential(Linear):
    @typechecker
    def __init__(
        self, layers: List[Union[Linear, Activation]], activation: Callable
    ) -> None:
        self.layers = layers
        self.activation = activation
        self._parameters = []
        self._grad = []
        for layer in layers:
            self._parameters += layer.parameters()
            self._grad += layer.grad()

    def parameters(self):
        return self._parameters

    def grad(self):
        return self._grad

    def forward(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        for linear in self.layers:
            output = linear.forward(input)
            input = output
        return output

    def __call__(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        return self.forward(input)

    def predict(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        return super().predict(input)

    def backward(self, dout: Float[torch.Tensor, "batch out_features"]) -> None:
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                layer.backward(dout)
                dout, _, _ = _linear_backward(dout, layer.cache)
            elif isinstance(layer, Activation):
                dout = layer.backward(dout)

    def fit(
        self,
        input: Float[torch.Tensor, "batch in_features"],
        target: Float[torch.Tensor, "batch out_features"],
        criterion: Loss,
        optimizer: Optimizer,
        n_epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        super().fit(input, target, criterion, optimizer, n_epochs, verbose)
