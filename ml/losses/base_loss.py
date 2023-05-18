from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import torch
from jaxtyping import Float

from ..metrics import functional as FM


class Loss(ABC):
    kwargs = dict(return_grad=True)
    
    @ abstractmethod
    def __init__(self, name: str, reduction: str = FM.DEFAULT_REDUCTION) -> None:
        super().__init__()
        self.kwargs.update(dict(reduction=reduction))
        self.loss_fn = partial(
            getattr(FM, name),
            **self.kwargs
        )
    
    def __call__(
        self, 
        output: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "batch out_features"]]:
        return self.loss_fn(output, target)
        