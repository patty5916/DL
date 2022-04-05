import pandas as pd
import numpy as np
from collections import Counter
import cv2
import time
import matplotlib.pyplot as plt

def load_data(PATH_TO_DESIRED_LOCATION):
    '''
    [Input]
      1. <string> PATH_TO_DESIRED_LOCATION: It should be the directory containing
      (1) images (2) train.txt (3) test.txt (4) val.txt

    [Output]
      1. <ndarray> np_train_txt: It contains both the directory to a specific image and the related label.
      2. <ndarray> np_test_txt: It contains both the directory to a specific image and the related label.
      3. <ndarray> np_val_txt: It contains both the directory to a specific image and the related label.
    '''

    # test.txt
    test_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION + "test.txt", sep=" ")
    np_test_txt = np.array(test_txt)

    # train.txt
    train_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION + "train.txt", sep=" ")
    np_train_txt = np.array(train_txt)

    # val.txt
    val_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION + "val.txt", sep=" ")
    np_val_txt = np.array(val_txt)

    print(f"[Check] There are {np_train_txt.shape[0]} pairs in train.txt.")
    print(f"[Check] There are {np_test_txt.shape[0]} pairs in test.txt.")
    print(f"[Check] There are {np_val_txt.shape[0]} pairs in val.txt.\n")

    return np_train_txt, np_test_txt, np_val_txt


def rgb_histogram(img):
    '''
    [Input]
      1. <ndarray> img: expected a square image.

    [Output]
      1. <list> X: a list contains 769 (256*3 + bias) elements.
    '''

    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    channels = [b, g, r]

    X = [1]  # bias

    for channel in channels:
        cnt = Counter(channel.reshape(img.shape[0] * img.shape[0]))

        for key in range(256):
            if key not in cnt.keys():
                X.append(0)
            else:
                X.append(cnt[key])

    return X


def create_COH_dataset(PATH_TO_DESIRED_LOCATION, metadata_file, image_size):
    '''
    [Input]
      1. <string> metadata_file: train.txt, val.txt, test.txt
      2. <int> image_size: each image has different size originally, so I convert each image to a fix size.

    [Output]
      1. <list> COH_Dataset: the data in COH_Dataset are a 1-D array containing 769 elements. 256*3+1=769
      2. <list> COH_Label: the label of each data.

    For each image, we do the following things:
      1. Read a specific image in RGB format, and also read the label.
      2. Resize the image to a fixed size. (ex: 256,256)
      3. Represent the image in form of Color of Histogram. (Also add a bias term at this step)
      4. Append the 1-D array into the COH_Dataset
      5. Append the label into the COH_Label
    '''
    COH_Dataset = []
    COH_Label = []
    counter = 0

    tic = time.time()
    for one_pair in metadata_file:
        # 1. Read a specific image in RGB format.
        img = cv2.imread(PATH_TO_DESIRED_LOCATION + one_pair[0])
        img_label = one_pair[1]

        # 2. Resize the image to a fixed size.
        img_resize = cv2.resize(img, (image_size, image_size))

        # 3. Represent the image in form of Color of Histogram.
        COH_array = rgb_histogram(img_resize)

        # 4. Append the 1-D array into the COH_Dataset.
        COH_Dataset.append(COH_array)

        # 5. Append the label into the COH_Label.
        COH_Label.append(img_label)

        # 6. Get to know the progress.
        counter = counter + 1
        if counter % 1000 == 0:
            toc = time.time()
            print(f'[Note] I have finished transforming {counter} images into other space.')
            print(f'This time, it takes {round(toc - tic, 2)} seconds."')
            tic = time.time()
    return COH_Dataset, COH_Label


def save_dataset(Dataset, Label, dataset_name, label_name):
    '''
    [Input]
      1. <list> Dataset: Train_COH_Dataset, Val_COH_Dataset, Test_COH_Dataset
      2. <string> desired_name:

    [Output] None
    '''

    tic = time.time()
    dataset = np.asarray(Dataset)
    np.savetxt(dataset_name, dataset, delimiter=',')

    label = np.asarray(Label)
    np.savetxt(label_name, label, delimiter=',')
    toc = time.time()

    print(f'[Save successfully]')
    print(f'It takes {round(toc - tic, 2)} seconds to save the {dataset_name} and the {label_name}.')


def CrossEntropy(y_pred, y_true):
    return (-1 * y_true * np.log(y_pred)).sum()


def Softmax(Z):
    S = np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)))
    return S


def get_top_5_predictions(Y_pred):
    '''
    [Input]
      1. <ndarray> Y_pred: It's a 1-D ndarray which contains the possibilities of the predictions.

    [Output]
      1. <int> top_1: The 1st likely breed among those 50 breeds.
      2. <int> top_2: The 2nd likely breed among those 50 breeds.
      3. <int> top_3: The 3rd likely breed among those 50 breeds.
      4. <int> top_4: The 4th likely breed among those 50 breeds.
      5. <int> top_5: The 5th likely breed among those 50 breeds.
    '''
    top_1 = Y_pred.argmax()
    Y_pred[0][top_1] = 0

    top_2 = Y_pred.argmax()
    Y_pred[0][top_2] = 0

    top_3 = Y_pred.argmax()
    Y_pred[0][top_3] = 0

    top_4 = Y_pred.argmax()
    Y_pred[0][top_4] = 0

    top_5 = Y_pred.argmax()

    return top_1, top_2, top_3, top_4, top_5


def topk_accuracy(Dataset, Label, W, Scale, Name):
    '''
    [Input]
      1. <list> Dataset: Either Train_COH_Dataset, Val_COH_Dataset, or Test_COH_Dataset
      2. <list> Label: Train_COH_Label, Val_COH_Label, Test_COH_Label
      3. <ndarry> W: It's the updated Weight from training process.
      4. <float> Scale: It's the hyper parameter we decided in training process.
      5. <String> Name: It's for convenient purpose

    [Output]
      1. <int> top1_accuracy
      2. <int> top5_accuracy
    '''
    num_top1_pred = 0
    num_top5_pred = 0
    len_dataset = len(Label)

    for i in range(len_dataset):

        # 1. Get the i-th data
        X = np.array(Dataset[i]).reshape(1, 769) / Scale
        Y = Label[i]
        assert X.shape == (1, 769), f'[Error] shape of X is {X.shape}. Expected shape is (1, 769).'

        # 2. Compute Z by using X and W.
        Z = np.dot(X, W)
        assert W.shape == (769, 50), f'[Error] shape of W is {W.shape}. Expected shape is (769, 50).'
        assert Z.shape == (1, 50), f'[Error] shape of Z is {X.shape}. Expected shape is (1, 50).'

        # 3. Predict the label by using Softmax.
        Y_pred = Softmax(Z)
        assert Y_pred.shape == (1, 50), f'[Error] shape of predict Y is {Y_pred.shape}. Expected shape is (1, 50).'

        # 4. Get top 5 predictions.
        top_1, top_2, top_3, top_4, top_5 = get_top_5_predictions(Y_pred)

        # 5. Check if the label is the top 1 prediction.
        if Y == top_1: num_top1_pred = num_top1_pred + 1

        # 6. Check if the label is in the top 5 predictions
        if Y in [top_1, top_2, top_3, top_4, top_5]: num_top5_pred = num_top5_pred + 1

    top1_accuracy = round(num_top1_pred / len_dataset * 100, 2)
    top5_accuracy = round(num_top5_pred / len_dataset * 100, 2)
    print(f'[Result of {Name}] The top-1 accuracy is {top1_accuracy} %')
    print(f'[Result of {Name}] The top-5 accuracy is {top5_accuracy} %')
    return top1_accuracy, top5_accuracy


def draw_train_val(Trian_Accuracy_top_1, Train_Accuracy_top_5, Val_Accuracy_top_1, Val_Accuracy_top_5):
    '''
    [Input]
      1. <list> Accuracy_top_1
      2. <list> Acciracy_top_5

    [Output] None
    '''

    plt.figure(figsize=(20, 10))
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Accuracy %", fontsize=20)

    plt.plot(Trian_Accuracy_top_1, label="Train Top-1")
    plt.plot(Train_Accuracy_top_5, label="Train Top-5")
    plt.plot(Val_Accuracy_top_1, label="Val Top-1")
    plt.plot(Val_Accuracy_top_5, label="Val Top-5")
    plt.legend(loc=2, fontsize=20)
    plt.show()


def draw_loss(E):
    '''
    [Input]
        1. <list> E

    [Output] None
    '''
    plt.figure(figsize=(20, 10))
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Loss: Cross Entropy", fontsize=20)
    plt.plot(E)
    plt.legend(loc=2, fontsize=20)
    plt.show()