from abc import ABC, abstractmethod
from typing import Callable, List

import torch

from .functional import identity


class Optimizer(ABC):
    n_steps: int = 0
    @abstractmethod
    def __init__(
        self, 
        paramters: List[torch.Tensor],
        grad: List[torch.Tensor],
        lr: float, 
        update_fn: Callable = identity, 
        **kwargs,
    ) -> None:
        super().__init__()
        self.parameters = paramters
        self.grad = grad
        self.lr = lr
        self.update_fn = update_fn
    
    def zero_grad(self) -> None:
        for i in range(len(self.grad)):
            self.grad[i] *= 0.0
    
    def step(self) -> None:
        for i, (param, grad) in enumerate(zip(self.parameters, self.grad)):
            self.parameters[i] = self.update_fn(param, grad, self.lr, self.n_steps)
        self.n_steps += 1
            