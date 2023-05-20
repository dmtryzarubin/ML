import torch
from sklearn.datasets import make_circles

from ml.activations import ReLU, Sigmoid, sigmoid
from ml.losses import BCELoss
from ml.models.linear import Linear
from ml.models.nn import Sequential

NUM_SAMPLES: int = 100
NOISE: float = 0.05
STATE: int = 2023
DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.float


def __create_data():
    X, y = make_circles(
        n_samples=NUM_SAMPLES,
        noise=NOISE,
        random_state=STATE,
    )
    X = torch.from_numpy(X).to(DTYPE)
    y = torch.from_numpy(y).to(DTYPE)
    if y.ndim == 1:
        y = y[..., None]
    return X, y


def __init_torch_linear(model: Sequential):
    t_model = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )
    for t, m in zip(t_model, model.layers):
        if isinstance(t, torch.nn.Linear):
            t.weight.data = m.weight.data.clone()
            t.bias.data = m.bias.data.clone()
    return t_model


def test_gradients():
    X, y = __create_data()

    model = Sequential([Linear(2, 16), ReLU(), Linear(16, 1)], activation=sigmoid)
    criterion = BCELoss()
    output = model(X)
    loss, dout = criterion(output, y)
    model.backward(dout)

    torch_model = __init_torch_linear(model)
    torch_output = torch_model(X)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(torch_output, y)
    torch_loss.backward()

    assert torch.allclose(output, torch_output)
    assert torch.allclose(loss, torch_loss)

    for t, l in zip(torch_model, model.layers):
        if isinstance(t, torch.nn.Linear) and isinstance(l, Linear):
            assert torch.allclose(t.weight.data, l.weight.data)
            assert torch.allclose(t.bias.data, l.bias.data)
