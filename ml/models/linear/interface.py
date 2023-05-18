from random import randint
from typing import Any, List, Optional

import torch
from beartype import beartype as typechecker
from jaxtyping import Float
from tqdm.notebook import tqdm

from ... import losses
from ...optim import Optimizer
from ..base_model import Model
from . import functional as F

__all__ = ["Linear"]


class Linear(Model):
    @typechecker
    def __init__(
        self, in_features: int, out_features: int, criterion: losses.Loss, **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.rand(out_features, in_features, **kwargs)
        self.bias = torch.rand(out_features, **kwargs) * 0.0
        self.d_weight = torch.zeros_like(self.weight)
        self.d_bias = torch.zeros_like(self.bias)

        self._parameters = [self.weight, self.bias]
        self._grad = [self.d_weight, self.d_bias]

        self.criterion = criterion

    def grad(self) -> List[torch.Tensor]:
        return self._grad

    def forward(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        output, self.cache = F._linear_forward(input, *self._parameters)
        return output

    def __call__(
        self, input: Float[torch.Tensor, "batch in_features"]
    ) -> Float[torch.Tensor, "batch out_features"]:
        output, self.cache = F._linear_forward(input, *self._parameters)
        return output

    def backward(self, dout: Float[torch.Tensor, "batch out_features"]) -> None:
        grad = F._linear_backward(dout, self.cache)
        for i, g in enumerate(grad):
            self._grad[i] += g

    def predict(
        self,
        input: Float[torch.Tensor, "batch in_features"],
    ) -> Float[torch.Tensor, "batch out_features"]:
        super().predict()
        return F.linear(input, *self._parameters)

    def fit(
        self,
        input: Float[torch.Tensor, "batch in_features"],
        target: Float[torch.Tensor, "batch out_features"],
        optimizer: Optimizer,
        n_epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        self.history = torch.zeros(n_epochs)
        pbar = tqdm(range(n_epochs), disable=not verbose)
        for epoch in pbar:
            output = self.forward(input)
            optimizer.zero_grad()
            loss, dout = self.criterion(output, target)
            self.backward(dout)
            optimizer.step()

            self.history[epoch] = loss
        super().fit(input)
