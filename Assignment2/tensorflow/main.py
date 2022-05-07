import argparse
from memory_profiler import profile

from tensorflow import keras
from keras import losses
from keras.callbacks import ModelCheckpoint

from utils import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser(description='For DL Assignment2')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--train_batch', type=int, default=1024, help='batch of training set')

    return parser.parse_args()


@profile(stream=open('tensorflow_LeNet5.log', 'w+'))
def main():
    args = parse_args()

    # data
    train_data = load_data('train').batch(args.train_batch)
    val_data = load_data('val').batch(1)
    test_data = load_data('test').batch(1)

    model = LeNet5()
    optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.8)
    model.compile(optimizer=optimizer, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    model_path = 'model.hdf5'
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbackList = [checkpoint]

    # train
    history = model.fit(train_data, epochs=args.epochs, callbacks=callbackList, validation_data=val_data)

    # test
    loss, acc = model.evaluate(test_data)
    print(f'Test Accuracy: {acc}')


    # plot
    plot_acc(history, 'tensorflow_LeNet5_acc.png')
    plot_loss(history, 'tensorflow_LeNet5_loss.png')


if __name__ == '__main__':
    main()