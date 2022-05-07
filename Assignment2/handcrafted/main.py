import argparse
import timeit
from memory_profiler import profile

from utils import *
from loss import *
from model import *
from optimizer import *


def parse_args():
    parser = argparse.ArgumentParser(description='For DL Assignment2')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--train_batch', type=int, default=128, help='batch of training set')
    parser.add_argument('--val_batch', type=int, default=128, help='batch of validation set')

    return parser.parse_args()


@profile(stream=open('handcrafted_LeNet5.log', 'w+'))
def main():
    args = parse_args()
    model = LeNet5()
    size = 128
    val_x, val_y = load_data('val', batch=args.val_batch, size=size)
    val_y = np.argmax(val_y, axis=1).reshape((len(val_y), 1))

    lossList = []
    train_accList = []
    val_accList = []
    epochsList = []
    best_acc = 0
    optimizer = SGDMomentum(model.get_parameters(), lr=args.lr, rho=0.8, reg=0.00003)
    criterion = CrossEntropyLoss()

    total_time = 0
    # train
    for epoch in range(args.epochs):
        start = timeit.default_timer()
        X_batch, Y_batch = load_data('train', batch=args.train_batch, size=size)
        Y_pred = model(X_batch)

        loss, dout = criterion.get(Y_pred, Y_batch)
        model.backward(dout)
        optimizer.step()

        Y_pred = Y_pred.argsort(axis=1)[:, -1:]
        Y_batch = np.argmax(Y_batch, axis=1).reshape((len(Y_batch), 1))
        train_acc = (Y_pred == Y_batch).mean()
        stop = timeit.default_timer()

        lossList.append(loss)
        print(f'Epochs: {epoch}\t|Loss: {loss}\t|Acc: {train_acc}\t|Time: {stop - start}')

        # validation
        if (epoch % 10) == 0:
            val_y_pred = model(val_x)
            val_y_pred = val_y_pred.argsort(axis=1)[:, -1:]
            val_acc = (val_y_pred == val_y).mean()
            print(f'Epochs: {epoch}\t|Loss: {loss}\t|Train Acc: {train_acc}\t|Val Acc: {val_acc}\t|Time: {stop - start}')

            train_accList.append(train_acc)
            val_accList.append(val_acc)
            epochsList.append(epoch)

            if val_acc > best_acc:
                model.save_parameters('handcrafted_LeNet5.npy')
                best_acc = val_acc


        total_time += (stop - start)

    # test
    model = LeNet5()
    model.load_parameters('handcrafted_LeNet5.npy')
    test_x, test_y = load_data('val', size=size)
    test_y = np.argmax(test_y, axis=1).reshape((len(test_y), 1))
    test_y_pred = model(test_x)
    test_y_pred = test_y_pred.argsort(axis=1)[:, -1:]
    test_acc = (test_y_pred == test_y).mean()
    print(f'Test Accuracy: {test_acc}')
    print(f'Total Time: {total_time}')

    # plot
    plot_acc(train_accList, val_accList, epochsList, 'handcrafted_LeNet5_acc.png')
    plot_loss(lossList, 'handcrafted_LeNet5_loss.png')


if __name__ == '__main__':
    main()