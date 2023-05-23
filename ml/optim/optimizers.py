from typing import List, Tuple

import torch

from ml.optim.functional import adam, sgd

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


class Adam(Optimizer):
    def __init__(
        self,
        parameters: List[torch.Tensor],
        grad: List[torch.Tensor],
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        **kwargs,
    ) -> None:
        self.ms = [torch.zeros_like(x) for x in grad]
        self.vs = [torch.zeros_like(x) for x in grad]
        self.betas = betas
        super().__init__(parameters, grad, lr, weight_decay, update_fn=adam, **kwargs)

    def step(self) -> None:
        for i, (param, grad, m, v) in enumerate(
            zip(self.parameters, self.grad, self.ms, self.vs)
        ):
            self.parameters[i], self.ms[i], self.vs[i] = self.update_fn(
                param=param,
                grad=grad,
                m=m,
                v=v,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
            self.n_steps += 1
