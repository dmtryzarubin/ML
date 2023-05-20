from typing import Callable, List

import torch
from beartype import beartype as typechecker
from jaxtyping import Float
from tqdm.autonotebook import tqdm

from ml.activations import identity

from ... import losses
from ...activations import identity, sigmoid
from ...optim import Optimizer
from ..base_model import Model
from . import functional as F

__all__ = ["Linear", "LinearRegression", "LogisticRegression"]


class Linear(Model):
    """
    Container with forward, backward, fit api for linear(Wx+b) transformation.
    """

    @typechecker
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable = identity,
        **kwargs
    ) -> None:
        """
        Container with forward, backward, fit api for linear(Wx+b) transformation.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param activation: Activation function that is used only for prediction.
        In case of Logistic regression can be used a sigmoid activation,
        but the model will return logits on forward pass, defaults to identity
        :param kwargs: Kwargs that are passed to `torch.randn`
        for weights and biases initialization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.rand(out_features, in_features, **kwargs)
        self.bias = torch.rand(out_features, **kwargs) * 0.0
        self.d_weight = torch.zeros_like(self.weight)
        self.d_bias = torch.zeros_like(self.bias)

        self._parameters = [self.weight, self.bias]
        self._grad = [self.d_weight, self.d_bias]
        self.activation = activation

    def grad(self) -> List[torch.Tensor]:
        return self._grad

    def forward(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        """
        Forward pass of affine transform. Also saves cached variables in `self.cache`

        :return: Output of Affine transformation
        """
        output, self.cache = F._linear_forward(input, *self._parameters)
        return output

    def __call__(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        """
        Forward pass of affine transform. Also saves cached variables in `self.cache`

        :return: Output of Affine transformation
        """
        return self.forward(input)

    def backward(self, dout: Float[torch.Tensor, "batch out_features"]) -> None:
        """
        Calculates and adds gradients for model parameters
        """
        d_input, d_weight, d_bias = F._linear_backward(dout, self.cache)
        grad = [d_weight, d_bias]
        for i, g in enumerate(grad):
            self._grad[i] += g

    def predict(
        self,
        input: Float[torch.Tensor, "batch in_features"],
    ) -> Float[torch.Tensor, "batch out_features"]:
        """
        Predicts target for input data

        :return: tensor with predictions of shape `{batch, out_features}`
        """
        super().predict()
        return self.activation(self.forward(input))

    def fit(
        self,
        input: Float[torch.Tensor, "batch in_features"],
        target: Float[torch.Tensor, "batch out_features"],
        criterion: losses.Loss,
        optimizer: Optimizer,
        n_epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Method for model fitting

        :param input: Train data
        :param target: Target variable
        :param criterion: E.g. loss that will be optimized
        :param optimizer: Optimizer that will update parameters due to some rule
        :param n_epochs: Number of iterations to perform, defaults to 100
        :param verbose: If True the tqdm bar will be used to disply progress, defaults to True
        """
        self.history = torch.zeros(n_epochs)
        pbar = tqdm(range(n_epochs), disable=not verbose)
        for epoch in pbar:
            output = self.forward(input)
            optimizer.zero_grad()
            loss, dout = criterion(output, target)
            self.backward(dout)
            optimizer.step()

            self.history[epoch] = loss
        super().fit(input)


class LogisticRegression(Linear):
    """
    Container for logistic regression model
    """

    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__(in_features, out_features, sigmoid, **kwargs)


class LinearRegression(Linear):
    """
    Container for linear regression model
    """

    pass
