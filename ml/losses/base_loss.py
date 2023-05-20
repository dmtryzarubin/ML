from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Tuple

import torch
from jaxtyping import Float

from ..metrics import functional as FM


class Loss(ABC):
    kwargs = dict(return_grad=True)

    @abstractmethod
    def __init__(
        self, loss_fn: Callable, reduction: str = FM.DEFAULT_REDUCTION
    ) -> None:
        """
        Base class for loss computation.

        :param loss_fn: Callable object,
        that must return loss due to reduction and compute gradient dL/doutput
        :param reduction: Loss reduction, 'mean' means that,
        loss will be averaged among features and items,
        defaults to FM.DEFAULT_REDUCTION
        """
        super().__init__()
        self.kwargs.update(dict(reduction=reduction))
        self.loss_fn = partial(loss_fn, **self.kwargs)

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "batch out_features"]]:
        """
        Calculates loss value and gradient

        :return: Loss value (maybe a scalar due to reduction)
        and gradient of shape {batch, num_features}
        """
        return self.loss_fn(output, target)
