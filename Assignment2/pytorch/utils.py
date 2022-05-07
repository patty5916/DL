import os
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_name, transform=None):
        super().__init__()
        self.transform = transform
        with open(f'../{data_name}.txt') as f:
            self.data = [content.split(' ') for content in f.read().split('\n')][:-1]
            f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = cv2.imread(f'../{self.data[item][0]}')
        label = self.data[item][1]

        if img.ndim != 3:
            # grayscale
            img_rgb = np.zeros([img.shape[0], img.shape[1], 3])
            img_rgb[:] = img.reshape(img.shape[0], img.shape[1], 1)
        else:
            # color: bgr -> rgb
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)

        return img_rgb, int(label)


def plot_acc(train_acc, val_acc, epochs, save_name):
    if os.path.isdir('plot'):
        pass
    else:
        os.mkdir('plot')
    fig = plt.figure()
    ax = sns.lineplot(epochs, train_acc, color='royalblue', label='train')
    sns.lineplot(epochs, val_acc, color='orange', label='validation')
    ax.set_xlabel('Epochs', size=14)
    ax.set_ylabel('Accuracy', size=14)
    ax.set_title('Top-1 Accuracy', size=14, fontweight='bold')
    ax.legend()
    fig.set_figheight(6)
    fig.set_figwidth(16)
    fig.savefig(f'plot/{save_name}')
    print('Save Top-1 accuracy successfully.')


def plot_loss(loss, save_name):
    if os.path.isdir('plot'):
        pass
    else:
        os.mkdir('plot')
    fig = plt.figure()
    plt.plot(loss, color='royalblue')
    plt.xlabel('epochs', size=14)
    plt.ylabel('loss', size=14)
    fig.savefig(f'plot/{save_name}')
    print('Save loss successfully.')
