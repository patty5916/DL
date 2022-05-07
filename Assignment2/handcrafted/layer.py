import numpy as np

from utils import img2color, color2img


class Linear():
    '''
    Fully connected layer
    '''

    def __init__(self, D_in, D_out):
        self.cache = None
        self.W = {'val': np.random.normal(0, np.sqrt(2 / D_in), (D_in, D_out)), 'grad': 0}
        self.b = {'val': np.random.randn(D_out), 'grad': 0}

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = np.dot(X, self.W['val'] + self.b['val'])
        self.cache = X
        return out

    def backward(self, dout):
        X = self.cache
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = np.dot(np.reshape(X, (X.shape[0], -1)).T, dout)
        self.b['grad'] = np.sum(dout, axis=0)
        return dX

    def update_parameters(self, lr=1e-3):
        self.W['val'] -= lr * self.W['grad']
        self.b['val'] -= lr * self.b['grad']


class Conv():
    def __init__(self, C_in, C_out, F, stride=1, padding=0, bias=True):
        self.stride = stride
        self.padding = padding
        self.W = {'val': np.random.normal(0, np.sqrt(2 / C_in), (C_out, C_in, F, F)), 'grad': 0}
        self.b = {'val': np.random.randn(C_out), 'grad': 0}

        self.x = None
        self.color = None
        self.color_W = None

        self.dW = None
        self.db = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, x):
        FN, C, FH, FW = self.W['val'].shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.padding - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.padding - FW) / self.stride)

        self.x = x
        self.color = img2color(x, FH, FW, self.stride, self.padding)
        self.color_W = self.W['val'].reshape(FN, -1).T

        out = np.dot(self.color, self.color_W) + self.b['val']
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W['val'].shape
        dout = dout.transpose(0, 2, 3, 1).reshape((-1, FN))

        self.W['grad'] = np.dot(self.color.T, dout)
        self.W['grad'] = self.W['grad'].transpose(1, 0).reshape(FN, C, FH, FW)
        self.b['grad'] = np.sum(dout, axis=0)

        dcolor = np.dot(dout, self.color_W.T)
        dx = color2img(dcolor, self.x.shape, FH, FW, self.stride, self.padding)
        return dx


class MaxPooling():
    '''
    Max Pooling
    '''

    def __init__(self, f, stride=2, padding=0):
        self.pooling_h = f
        self.pooling_w = f
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pooling_h) / self.stride)
        out_w = int(1 + (W - self.pooling_w) / self.stride)

        color = img2color(x, self.pooling_h, self.pooling_w, self.stride, self.padding)
        color = color.reshape(-1, self.pooling_h * self.pooling_w)

        self.x = x
        self.arg_max = np.argmax(color, axis=1)
        out = np.max(color, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pooling_size = self.pooling_h * self.pooling_w
        dmax = np.zeros((dout.size, pooling_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pooling_size,))

        dcolor = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = color2img(dcolor, self.x.shape, self.pooling_h, self.pooling_w, self.stride, self.padding)
        return dx
