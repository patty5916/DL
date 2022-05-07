from abc import ABCMeta, abstractmethod

from layer import *
from activation import *


class Net(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        pass


class LeNet5(Net):
    def __init__(self):
        self.conv1 = Conv(3, 6, 5)
        self.sigmoid1 = Sigmoid()
        self.pool1 = MaxPooling(2, 2)
        self.conv2 = Conv(6, 16, 5)
        self.sigmoid2 = Sigmoid()
        self.pool2 = MaxPooling(2, 2)
        self.fc1 = Linear(29 * 29 * 16, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 50)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, x):
        h1 = self.conv1(x)
        a1 = self.sigmoid1(h1)
        p1 = self.pool1(a1)
        h2 = self.conv2(p1)
        a2 = self.sigmoid2(h2)
        p2 = self.pool2(a2)
        self.p2_shape = p2.shape
        f1 = p2.reshape(x.shape[0], -1)  # flatten
        h3 = self.fc1(f1)
        h4 = self.fc2(h3)
        h = self.fc3(h4)
        return h

    def backward(self, dout):
        dout = self.fc3.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.fc1.backward(dout)
        dout = dout.reshape(self.p2_shape)
        dout = self.pool2.backward(dout)
        dout = self.sigmoid2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.sigmoid1.backward(dout)
        dout = self.conv1.backward(dout)

    def get_parameters(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]

    def set_parameters(self, parameters):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b] = parameters

    def save_parameters(self, path):
        parameters = self.get_parameters()
        np.save(path, parameters)
        print('Save parameters successfully.')

    def load_parameters(self, path):
        parameters = np.load(path, allow_pickle=True)
        self.set_parameters(parameters)
        print('Load parameters successfully.')