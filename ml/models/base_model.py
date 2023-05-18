from abc import ABC, abstractmethod
from typing import List

import torch


class Model(ABC):
    _fit_called: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._parameters = []

    @abstractmethod
    def fit(self, input: torch.Tensor) -> None:
        self._fit_called = True

    @abstractmethod
    def predict(self) -> None:
        if not self._fit_called:
            raise UserWarning("Model hasn't been fitted yet.")

    @property
    def fit_called(self) -> bool:
        return self._fit_called

    def parameters(self) -> List[torch.Tensor]:
        return self._parameters
