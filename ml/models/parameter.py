import torch


class Parameter(torch.Tensor):
    def __init__(self, *args) -> None:
        super().__init__()
        self.grad = torch.zeros_like(self.data)
