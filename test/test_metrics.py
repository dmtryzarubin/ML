import torch
from sklearn import metrics as skm

from ml import metrics

torch.manual_seed(2023)


def test_metrics():
    data = torch.randn(100, 2), torch.randn(100, 2)
    comparisons = [
        ["mse", "mean_squared_error"],
        ["mae", "mean_absolute_error"],
        ["r2_score", "r2_score"],
    ]
    reduction = "mean"
    for local_name, sk_name in comparisons:
        a = getattr(metrics, local_name)(*data, reduction=reduction)
        b = torch.tensor(getattr(skm, sk_name)(data[1], data[0])).to(torch.float)
        assert torch.allclose(a, b)
