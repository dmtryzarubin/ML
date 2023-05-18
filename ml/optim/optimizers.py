from typing import Callable

from ml.models import linear
from ml.optim.functional import sgd

from .base_optim import Optimizer


class SGD(Optimizer):
    def __init__(self, *args, **kwargs) -> None:
        update_fn = sgd
        super().__init__(*args, update_fn=update_fn, **kwargs)
