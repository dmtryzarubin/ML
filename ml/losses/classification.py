from ml.metrics import functional as FM

from .base_loss import Loss


class BCELoss(Loss):
    """
    Binary cross entropy with logits loss container
    """

    def __init__(self, reduction: str = FM.DEFAULT_REDUCTION) -> None:
        super().__init__(FM.binary_cross_entropy_with_logits, reduction)
