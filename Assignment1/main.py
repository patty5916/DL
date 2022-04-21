import argparse
import os
import numpy as np


from utils import *
from perceptron import *


def parse_args():
    parser = argparse.ArgumentParser(description='For DL Assignment1')
    parser.add_argument('-perceptron', type=str, help='one/two')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.perceptron == 'one':
        PATH_TO_DESIRED_LOCATION = 'D:/NCKU/110_2/DeepLearning/HW/HW1/'
        if os.path.isfile(PATH_TO_DESIRED_LOCATION + 'Train_COH_Dataset.csv'):
            # load data
            # Training Dataset
            Train_COH_Dataset = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Train_COH_Dataset.csv", header=None)
            Train_COH_Label = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Train_COH_Label.csv", header=None)

            # Validation Dataset
            Val_COH_Dataset = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Val_COH_Dataset.csv", header=None)
            Val_COH_Label = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Val_COH_Label.csv", header=None)

            # Testing Dataset
            Test_COH_Dataset = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Test_COH_Dataset.csv", header=None)
            Test_COH_Label = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Test_COH_Label.csv", header=None)
        else:
            # create data
            np_train_txt, np_test_txt, np_val_txt = load_data(PATH_TO_DESIRED_LOCATION)

            # Training Dataset
            Train_COH_Dataset, Train_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_train_txt, 256)
            save_dataset(Train_COH_Dataset, Train_COH_Label, 'Train_COH_Dataset.csv', 'Train_COH_Label.csv')

            # Validation Dataset
            Val_COH_Dataset, Val_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_val_txt, 256)
            save_dataset(Val_COH_Dataset, Val_COH_Label, 'Val_COH_Dataset.csv', 'Val_COH_Label.csv')

            # Testing Dataset
            Test_COH_Dataset, Test_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_test_txt, 256)
            save_dataset(Test_COH_Dataset, Test_COH_Label, 'Test_COH_Dataset.csv', 'Test_COH_Label.csv')

            np_Train_COH_Dataset = np.array(Train_COH_Dataset)
            np_Train_COH_Label = np.array(Train_COH_Label).reshape(len(Train_COH_Label), 1)
            np_Val_COH_Dataset = np.array(Val_COH_Dataset)
            np_Val_COH_Label = np.array(Val_COH_Label).reshape(len(Val_COH_Label), 1)
            print(f'The shape of np_Train_COH_Dataset is {np_Train_COH_Dataset.shape}')
            print(f'The shape of np_Train_COH_Label is {np_Train_COH_Label.shape}')

        # Initialize weight matrix W
        np.random.seed(0)
        W = np.random.uniform(low=-0.01, high=0.01, size=(769, 50))
        # Setup hyper parameters
        Epoch = 50
        r = 0.0003
        Scale = 1000

        # Accuracy_top_1 will contain the top-1 accuracy per epoch
        Trian_Accuracy_top_1 = []
        Val_Accuracy_top_1 = []

        # Accuracy_top_5 will contain the top-5 accuracy per epoch
        Train_Accuracy_top_5 = []
        Val_Accuracy_top_5 = []

        # E will contain the average cross-entropy per epoch
        E = []
        len_COH_Dataset = len(Train_COH_Label)
        tic = time.time()
        for epoch in range(Epoch):
            # shuffle the training dataset.
            random_index = np.arange(len_COH_Dataset)
            np.random.shuffle(random_index)

            # e will record errors in an epoch.
            e = []
            for i in random_index:
                # Grab the i-th training data.
                X = np.array(Train_COH_Dataset[i]).reshape(1, 769) / Scale
                assert X.shape == (1, 769), f'[Error] shape of X is {X.shape}. Expected shape is (1, 769)'

                # Compute Z by using X and W.
                Z = np.dot(X, W)
                assert W.shape == (769, 50), f'[Error] shape of W is {W.shape}. Expected shape is (769, 50).'
                assert Z.shape == (1, 50), f'[Error] shape of Z is {X.shape}. Expected shape is (1, 50).'

                # Predict the label by using Softmax
                Y_pred = Softmax(Z)
                assert Y_pred.shape == (1, 50), f'[Error] shape of predict Y is {Y_pred.shape}. Expected shape is (1, 50).'

                # Grab the i-th label of i-th training data.
                label = Train_COH_Label[i]
                Y_truth = np.zeros(50).reshape(1, 50)
                Y_truth[0][label] = 1
                assert Y_truth.shape == (1, 50), f'[Error] shape of true Y is {Y_truth.shape}. Expected shape is (1, 50).'

                # Record cross entropy error
                e.append(round(CrossEntropy(Y_pred, Y_truth), 4))

                # Compute dE/dZ. This term is related to the derivative of softmax function. Magically the "dEdZ" is equal to "Y_pred - Y_truth"
                dEdZ = Y_pred - Y_truth
                assert dEdZ.shape == (1, 50), f'[Error] shape of dEdZ is {dEdZ.shape}. Expected shape is (1, 50).'

                # Compute the gradients w.r.t. the weight matrix W.
                dW = np.outer(X, dEdZ)
                assert dW.shape == (769, 50), f'[Error] shape of dW is {dW.shape}. Expected shape is (769, 50).'

                # Update the parameters
                W = W - r * dW

            toc = time.time()
            print(f'\n[Training] Epoch: {epoch}\t|Loss: {round(np.mean(e), 4)}.\t|Time: {round(toc - tic, 2)} sec.')
            tic = time.time()

            # Measure the top-1 accuracy and top-5 accuracy
            train_top1_accuracy, train_top5_accuracy = topk_accuracy(Train_COH_Dataset, Train_COH_Label, W, Scale, 'Train')
            val_top1_accuracy, val_top5_accuracy = topk_accuracy(Val_COH_Dataset, Val_COH_Label, W, Scale, 'Val')
            test_top1_accuracy, test_top5_accuracy = topk_accuracy(Test_COH_Dataset, Test_COH_Label, W, Scale, 'Test')

            # Collect results
            E.append(np.mean(e))
            Trian_Accuracy_top_1.append(train_top1_accuracy)
            Train_Accuracy_top_5.append(train_top5_accuracy)
            Val_Accuracy_top_1.append(val_top1_accuracy)
            Val_Accuracy_top_5.append(val_top5_accuracy)


        # Evaluate
        draw_train_val(Trian_Accuracy_top_1, Train_Accuracy_top_5, Val_Accuracy_top_1, Val_Accuracy_top_5)
        draw_loss(E)

        print('[Final]')
        test_top1_accuracy, test_top5_accuracy = topk_accuracy(Test_COH_Dataset, Test_COH_Label, W, Scale, 'Test')

    elif args.perceptron == 'two':
        PATH_TO_DESIRED_LOCATION = 'D:/NCKU/110_2/DeepLearning/HW/HW1/'
        if os.path.isfile(PATH_TO_DESIRED_LOCATION + 'Train_COH_Dataset.csv'):
            # load data
            # Training Dataset
            train_X = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Train_COH_Dataset.csv", header=None)
            train_Y = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Train_COH_Label.csv", header=None)
            len_train_X = len(train_X)

            # Validation Dataset
            val_X = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Val_COH_Dataset.csv", header=None)
            val_Y = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Val_COH_Label.csv", header=None)

            # Testing Dataset
            test_X = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Test_COH_Dataset.csv", header=None)
            test_Y = pd.read_csv(PATH_TO_DESIRED_LOCATION + "Test_COH_Label.csv", header=None)
        else:
            # create data
            np_train_txt, np_test_txt, np_val_txt = load_data(PATH_TO_DESIRED_LOCATION)

            # Training Dataset
            train_X, train_Y = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_train_txt, 256)
            save_dataset(train_X, train_Y, 'Train_COH_Dataset.csv', 'Train_COH_Label.csv')
            len_train_X = len(train_X)

            # Validation Dataset
            val_X, val_Y = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_val_txt, 256)
            save_dataset(val_X, val_Y, 'Val_COH_Dataset.csv', 'Val_COH_Label.csv')

            # Testing Dataset
            test_X, test_Y = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_test_txt, 256)
            save_dataset(test_X, test_Y, 'Test_COH_Dataset.csv', 'Test_COH_Label.csv')

            np_train_X = np.array(train_X)
            np_train_Y = np.array(train_Y).reshape(len(train_Y), 1)
            np_val_X = np.array(val_X)
            np_val_Y = np.array(val_Y).reshape(len(val_Y), 1)
            print(f'The shape of np_Train_COH_Dataset is {train_X.shape}')
            print(f'The shape of np_Train_COH_Label is {train_Y.shape}')


        # Initialize the Custom Class
        NN = nn()

        # Initialize Weight Matrix
        W1, W2 = NN.initialize_weights()

        # Accuracy_top_1 and Accuracy_top_5 will record the top-1 and top-5 accuracies. And E will record the loss.
        Trian_Accuracy_top_1, Val_Accuracy_top_1, Train_Accuracy_top_5, Val_Accuracy_top_5, E = [], [], [], [], []

        # Hyper parameters
        Epoch = 100
        lr_1, lr_2 = 0.01, 0.01
        Scale = 100.0


        tic = time.time()
        for epoch in range(Epoch):

            # Shuffle the training dataset.
            random_index = np.arange(len_train_X)
            np.random.shuffle(random_index)

            e = []
            for i in random_index:
                # Get the i-th training data.
                X = np.array(train_X[i:i + 1]).reshape(1, 769) / Scale

                # Forward Propagation.
                Y_pred, A2, A1 = NN.forward_pass(X, W1, W2)

                # Get the i-th label.
                label = int(train_Y[i:i + 1][0])
                Y_truth = np.zeros(50).reshape(1, 50)
                Y_truth[0][label] = 1
                assert Y_truth.shape == (1, 50), f'[Error] shape of truth Y is {Y_truth.shape}. Expected shape is (1, 50).'

                # Record cross entropy.
                e.append(round(NN.cross_entropy(Y_pred, Y_truth), 4))

                # Backward Propagation.
                dEdW1, dEdW2 = NN.backward_pass(Y_pred, Y_truth, A2, A1, X, W2, W1)

                # Update Weight Matrix.
                if (epoch < 50):
                    W1, W2 = NN.update_weights(dEdW1, dEdW2, W1, W2, lr_1)
                else:
                    W1, W2 = NN.update_weights(dEdW1, dEdW2, W1, W2, lr_2)

            toc = time.time()
            print(f'\n[Training] Epoch: {epoch}\t|Loss: {round(np.mean(e), 4)}\t|Time: {round(toc - tic, 2)} sec.')
            tic = time.time()

            # Measure the top-1 accuracy and top-5 accuracy
            train_top1_accuracy, train_top5_accuracy = NN.top_accuracy(train_X, train_Y, W1, W2, Scale, 'Train')
            val_top1_accuracy, val_top5_accuracy = NN.top_accuracy(val_X, val_Y, W1, W2, Scale, 'Val')

            # Collect results
            E.append(np.mean(e))
            Trian_Accuracy_top_1.append(train_top1_accuracy)
            Train_Accuracy_top_5.append(train_top5_accuracy)
            Val_Accuracy_top_1.append(val_top1_accuracy)
            Val_Accuracy_top_5.append(val_top5_accuracy)

        # Evaluate
        draw_train_val(Trian_Accuracy_top_1, Train_Accuracy_top_5, Val_Accuracy_top_1, Val_Accuracy_top_5)
        draw_loss(E)

        print('[Final]')
        test_top1_accuracy, test_top5_accuracy = NN.top_accuracy(test_X, test_Y, W1, W2, Scale, 'Test')

    else:
        print('Wrong number of perceptron layers.')



