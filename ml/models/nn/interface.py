from typing import Callable, List, Union

import torch
from beartype import beartype as typechecker
from jaxtyping import Float
from tqdm.notebook import tqdm

from ...activations import Activation, ReLU, Sigmoid, identity
from ...losses import Loss
from ...optim import Optimizer
from ..linear import Linear
from ..linear.functional import _linear_backward
from .init import kaiming_normal, kaiming_uniform


class Sequential(Linear):
    """
    Container for stack of sequential linear layers and activations
    """

    @typechecker
    def __init__(
        self,
        layers: List[Union[Linear, Activation]],
        activation: Callable = identity,
        init_fn: Callable = kaiming_normal,
    ) -> None:
        """
        Container for stack of sequential linear layers and activations

        :param layers: List of `Linear` and Activat
        :param activation: Activation function that is used after a final layer on logits
        """
        self.layers = layers
        self._init_weights(init_fn)
        self.activation = activation
        self._parameters = []
        self._grad = []
        for layer in self.layers:
            self._parameters += layer.parameters()
            self._grad += layer.grad()

    def _init_weights(self, init_fn):
        for i, layer in enumerate(self.layers):
            # assuming that we have a stack of [linear, non_linear, linear, ...., linear]
            if isinstance(layer, Activation) and i < len(self.layers) - 1:
                self.layers[i - 1].weight = init_fn(
                    self.layers[i - 1].weight, layer._name
                )
                self.layers[i - 1].bias *= 0.0
            if isinstance(layer, Linear) and i == len(self.layers) - 1:
                self.layers[i].weight *= 0.01

    def forward(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        """
        Performs forward pass for each layer and activation sequentially

        :return: Output with shape `{batch, out_features_of_last_layer}`
        """
        for linear in self.layers:
            output = linear.forward(input)
            input = output
        return output

    def __call__(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        """
        Performs forward pass for each layer and activation sequentially

        :return: Output with shape `{batch, out_features_of_last_layer}`
        """
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
