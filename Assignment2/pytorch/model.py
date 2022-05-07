import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(29 * 29 * 16, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 50)
        )

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h