from .functional import *

__all__ = ["Activation", "Sigmoid", "ReLU"]


class Activation:
    _parameters = []
    _grad = []

    def parameters(self):
        return self._parameters

    def grad(self):
        return self._grad

    def forward(self, input):
        return input

    def backward(self, dout):
        return dout


class Sigmoid(Activation):
    def forward(self, input):
        self.cache = sigmoid(input)
        return self.cache

    def backward(self, dout):
        return dout * self.cache * (1 - self.cache)


class ReLU(Activation):
    def forward(self, input):
        output = relu(input)
        self.cache = (output > 0).to(input)
        return output

    def backward(self, dout):
        return dout * self.cache
