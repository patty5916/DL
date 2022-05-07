import numpy as np

class SGD():
    def __init__(self, parameters, lr=1e-3, reg=0):
        self.parameters = parameters
        self.lr = lr
        self.reg = reg

    def step(self):
        for parameter in self.parameters:
            parameter['val'] -= (self.lr * parameter['val'] + self.reg * parameter['val'])


class SGDMomentum():
    def __init__(self, parameters, lr=1e-3, rho=0.99, reg=0):
        self.parameters = parameters
        self.n_parameters = len(parameters)
        self.lr = lr
        self.rho = rho
        self.reg = reg

        self.velocities = []
        for parameter in self.parameters:
            self.velocities.append(np.zeros(parameter['val'].shape))

    def step(self):
        for i in range(self.n_parameters):
            self.velocities[i] = self.rho * self.velocities[i] + (1 - self.rho) * self.parameters[i]['grad']
            self.parameters[i]['val'] -= (self.lr * self.velocities[i] + self.reg * self.parameters[i]['val'])
