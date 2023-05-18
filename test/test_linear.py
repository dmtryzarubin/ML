import torch
from sklearn.datasets import make_regression

from ml.losses import MAELoss
from ml.models.linear import Linear

NUM_SAMPLES: int = 100
NOISE: float = 5.0
BIAS: float = 1.23
STATE: int = 2023
IN_FEATURES: int = 5
OUT_FEATURES: int = 5
DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.float


def __create_data():
    X, y = make_regression(
        n_samples=NUM_SAMPLES,
        n_targets=OUT_FEATURES,
        n_features=IN_FEATURES,
        n_informative=IN_FEATURES,
        noise=NOISE,
        bias=BIAS,
        random_state=STATE,
        coef=False,
    )
    X = torch.from_numpy(X).to(DTYPE)
    y = torch.from_numpy(y).to(DTYPE)
    if y.ndim == 1:
        y = y[..., None]
    return X, y


def __init_torch_linear(w, b):
    torch_model = torch.nn.Linear(
        IN_FEATURES, OUT_FEATURES, bias=True, device=DEVICE, dtype=DTYPE
    )
    torch_model.weight = torch.nn.Parameter(w.clone(), requires_grad=True)
    torch_model.bias = torch.nn.Parameter(b.clone(), requires_grad=True)
    return torch_model


def test_gradients():
    X, y = __create_data()

    criterion = MAELoss()
    model = Linear(IN_FEATURES, OUT_FEATURES, criterion, device=DEVICE, dtype=DTYPE)
    output = model(X)
    loss, dout = criterion(output, y)
    model.backward(dout)

    torch_model = __init_torch_linear(model.weight, model.bias)
    torch_output = torch_model(X)
    torch_loss = torch.nn.functional.l1_loss(torch_output, y)
    torch_loss.backward()

    assert torch.allclose(torch_output, output)
    assert torch.allclose(torch_loss.data, loss.data)
    assert torch.allclose(torch_model.bias.grad, model.d_bias)
    assert torch.allclose(torch_model.weight.grad, model.d_weight)
