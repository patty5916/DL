import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize


def load_data(data_name, batch=False, size=62):
    mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
    std = np.array([[[0.229]], [[0.224]], [[0.225]]])

    if data_name == 'train':
        with open('../train.txt') as f:
            contents = [content.split(' ') for content in f.read().split('\n')][:-1]

            data = np.zeros([batch, 3, size, size])
            sample_contents = random.sample(contents, batch)
            label = []

            for index, path in enumerate(sample_contents):
                img = imread(f'../{path[0]}')
                img = resize(img, (size, size))
                img = np.array(img)

                if img.ndim != 3:
                    # grayscale
                    img_rgb = np.zeros([3, size, size])
                    img_rgb[:] = img.reshape(1, size, size)
                    img_rgb = (img_rgb - mean)
                    data[index, :] = img_rgb
                else:
                    # color: bgr -> rgb
                    img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
                    img = (img - mean)
                    data[index, :] = img

                img_label = np.zeros(50)
                img_label[int(path[1])] = 1
                label.append(img_label)

            f.close()
        return data, np.array(label)

    else:
        with open(f'../{data_name}.txt') as f:
            contents = [content.split(' ') for content in f.read().split('\n')][:-1]

            if batch:
                data = np.zeros([batch, 3, size, size])
                sample_contents = random.sample(contents, batch)
                label = []

                for index, path in enumerate(sample_contents):
                    img = imread(f'../{path[0]}')
                    img = resize(img, (size, size))
                    img = np.array(img)

                    if img.ndim != 3:
                        # grayscale
                        img_rgb = np.zeros([3, size, size])
                        img_rgb[:] = img.reshape(1, size, size)
                        img_rgb = (img_rgb - mean)
                        data[index, :] = img_rgb
                    else:
                        # color: bgr -> rgb
                        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
                        img = (img - mean)
                        data[index, :] = img

                    img_label = np.zeros(50)
                    img_label[int(path[1])] = 1
                    label.append(img_label)

            else:
                data = np.zeros([len(contents), 3, size, size])
                label = []

                for index, path in enumerate(contents):
                    img = imread(f'../{path[0]}')
                    img = resize(img, (size, size))
                    img = np.array(img)

                    if img.ndim != 3:
                        # grayscale
                        img_rgb = np.zeros([3, size, size])
                        img_rgb[:] = img.reshape(1, size, size)
                        img_rgb = (img_rgb - mean)
                        data[index, :] = img_rgb
                    else:
                        # color: bgr -> rgb
                        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
                        img = (img - mean)
                        data[index, :] = img

                    img_label = np.zeros(50)
                    img_label[int(path[1])] = 1
                    label.append(img_label)

            f.close()
        return data, np.array(label)


def color2img(color, input_shape, filter_h, filter_w, stride=1, padding=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1
    color = color.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
    for y in range(filter_h):
        max_y = y + stride * out_h
        for x in range(filter_w):
            max_x = x + stride * out_w
            img[:, :, y:max_y:stride, x:max_x:stride] += color[:, :, y, x, :, :]

    return img[:, :, padding:(H + padding), padding:(W + padding)]


def img2color(input_img, filter_h, filter_w, stride=1, padding=0):
    N, C, H, W = input_img.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = np.pad(input_img, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    color = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        max_y = y + stride * out_h
        for x in range(filter_w):
            max_x = x + stride * out_w
            color[:, :, y, x, :, :] = img[:, :, y:max_y:stride, x:max_x:stride]

    return color.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


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
