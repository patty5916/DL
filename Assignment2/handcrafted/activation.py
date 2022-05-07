import numpy as np

class ReLU():
    def __init__(self):
        self.mask = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid():
    def __init__(self):
        self.cache = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        X = np.clip(X, -50, 50)
        X = 1 / (1 + np.exp(-X))
        self.cache = X
        return X

    def backward(self, dout):
        X = self.cache
        dX = dout * X * (1 - X)
        return dX


class Softmax():
    def __init__(self):
        pass

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        X = X - np.max(X, axis=-1, keepdims=True)
        return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)