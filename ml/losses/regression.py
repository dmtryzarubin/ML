from ml.metrics import functional as FM

from .base_loss import Loss

__all__ = ["MSELoss", "MAELoss"]


class MSELoss(Loss):
    def __init__(self, reduction: str = FM.DEFAULT_REDUCTION) -> None:
        super().__init__("mse", reduction)


class MAELoss(Loss):
    def __init__(self, reduction: str = FM.DEFAULT_REDUCTION) -> None:
        super().__init__("mae", reduction)
