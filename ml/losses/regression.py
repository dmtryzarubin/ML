from ml.metrics import functional as FM

from .base_loss import Loss

__all__ = ["MSELoss", "MAELoss", "MAPELoss"]


class MSELoss(Loss):
    """
    Mean squared error loss container
    """

    def __init__(self, reduction: str = FM.DEFAULT_REDUCTION) -> None:
        super().__init__(FM.mse, reduction)


class MAELoss(Loss):
    """
    Mean absolute error loss container
    """

    def __init__(self, reduction: str = FM.DEFAULT_REDUCTION) -> None:
        super().__init__(FM.mae, reduction)


class MAPELoss(Loss):
    """
    Mean absolute percentage error loss container
    """

    def __init__(self, reduction: str = FM.DEFAULT_REDUCTION) -> None:
        super().__init__(FM.mape, reduction)
