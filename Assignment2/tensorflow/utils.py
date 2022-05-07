import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_images(files):
    label = tf.cast(tf.strings.to_number(files[1]), tf.int32)
    image = tf.io.read_file(files[0])
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    if len(image.shape) != 3:
        # grayscale
        image = tf.expand_dims(image, 2)
        image = (image - mean) / std
        image = tf.repeat(image, 3, 2)
        image = tf.image.resize(image, (128, 128))
    else:
        # color
        image = (image - mean) / std
        image = tf.image.resize(image, (128, 128))

    return image, label


def load_data(data_name):
    with open(f'../{data_name}.txt') as f:
        contents = [content.split(' ') for content in f.read().split('\n')][:-1]
        data = [[f'../{image_path}', label] for image_path, label in contents]
        f.close()

    random.shuffle(data)
    files = tf.data.Dataset.from_tensor_slices(data)
    images = files.map(load_images)

    return images


def plot_acc(history, save_name):
    if os.path.isdir('plot'):
        pass
    else:
        os.mkdir('plot')
    fig = plt.figure()
    plt.plot(history.history['accuracy'], color='royalblue')
    plt.plot(history.history['val_accuracy'], color='orange')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    fig.savefig(f'plot/{save_name}')
    print('Save accuracy successfully.')


def plot_loss(history, save_name):
    if os.path.isdir('plot'):
        pass
    else:
        os.mkdir('plot')
    fig = plt.figure()
    plt.plot(history.history['loss'], color='royalblue')
    plt.plot(history.history['val_loss'], color='orange')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    fig.savefig(f'plot/{save_name}')
    print('Save loss successfully.')
