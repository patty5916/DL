from activation import *
import numpy as np


class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax(Y_pred)
        loss = -np.sum(Y_true * np.log(prob + 1e-8)) / N
        Y_max = np.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[np.arange(N), Y_max] -= 1
        return loss, dout
