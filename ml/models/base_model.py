from abc import ABC, abstractmethod
from typing import List

import torch


class Model(ABC):
    """
    Base class for model with `.fit()` and `predict()` api.
    """

    _fit_called: bool = False

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self._parameters = []

    @abstractmethod
    def fit(self, input: torch.Tensor) -> None:
        """
        Method for model fitting
        """
        self._fit_called = True

    @abstractmethod
    def predict(self) -> None:
        """
        Predicts target for input data
        """
        if not self._fit_called:
            raise UserWarning("Model hasn't been fitted yet.")

    @property
    def fit_called(self) -> bool:
        return self._fit_called

    def parameters(self) -> List[torch.Tensor]:
        """
        Method that returns List of model parameters, stored in torch.Tensor

        :return: List of model parameters
        """
        return self._parameters
