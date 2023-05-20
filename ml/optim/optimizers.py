from typing import List

import torch

from ml.optim.functional import sgd

from .base_optim import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        parameters: List[torch.Tensor],
        grad: List[torch.Tensor],
        lr: float,
        weight_decay: float,
        **kwargs,
    ) -> None:
        super().__init__(parameters, grad, lr, weight_decay, update_fn=sgd, **kwargs)
