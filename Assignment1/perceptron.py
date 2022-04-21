import numpy as np

class nn():
    def __init__(self):
        pass

    def forward_pass(self, X, W1, W2):
        '''
        [Input]
          1. <ndarray> X: Its shape should be (1, 769).
          2. <ndarray> W1: Its shape should be (769, 300).
          3. <ndarray> W2: Its shape should be (300, 50).

        [Output]
          1. <ndarray> Y_pred: Its shape should be (1, 50).
          2. <ndarray> A2: Its shape should be (1, 50).
          3. <ndarray> A1: Its shape should be (1, 300).
        '''
        assert X.shape == (1, 769), f'[Error] shape of X is {X.shape}. Expected shape is (1, 769)'
        assert W1.shape == (769, 300), f'[Error] shape of W1 is {W1.shape}. Expected shape is (769, 300).'
        assert W2.shape == (300, 50), f'[Error] shape of W2 is {W2.shape}. Expected shape is (300, 50).'

        Z1 = np.dot(X, W1)
        assert Z1.shape == (1, 300), f'[Error] shape of Z1 is {Z1.shape}. Expected shape is (1, 300).'

        A1 = self.sigmoid(Z1)
        assert A1.shape == (1, 300), f'[Error] shape of A1 is {A1.shape}. Expected shape is (1, 300).'

        Z2 = np.dot(A1, W2)
        assert Z2.shape == (1, 50), f'[Error] shape of Z2 is {Z2.shape}. Expected shape is (1, 50).'

        A2 = self.sigmoid(Z2)
        assert A2.shape == (1, 50), f'[Error] shape of A2 is {A2.shape}. Expected shape is (1, 50).'

        Y_pred = self.softmax(A2)
        assert Y_pred.shape == (1, 50), f'[Error] shape of predict Y is {Y_pred.shape}. Expected shape is (1, 50).'

        return Y_pred, A2, A1


    def backward_pass(self, Y_pred, Y_truth, A2, A1, X, W2, W1):
        '''
        [Input]
          1. <ndarray> Y_pred: Its shape should be (1, 50).
          2. <ndarray> Y_truth: Its shape should be (1, 50).
          3. <ndarray> A2: Its shape should be (1, 50).
          4. <ndarray> A1: Its shape should be (1, 300).
          5. <ndarray> X: Its shape should be (1, 769).
          6. <ndarray> W2: Its shape should be (300, 50).
          7. <ndarray> W1: Its shape should be (769, 300).

        [Output]
          1. <ndarray> dEdW1: Its shape should be the same as W1, which is (769, 300).
          2. <ndarray> dEdW2: Its shape should be the same as W2, which is (300, 50).
        '''
        assert Y_pred.shape == (1, 50), f'[Error] shape of predict Y is {Y_pred.shape}. Expected shape is (1, 50).'
        assert Y_truth.shape == (1, 50), f'[Error] shape of truth Y is {Y_truth.shape}. Expected shape is (1, 50).'
        assert A2.shape == (1, 50), f'[Error] shape of A2 is {A2.shape}. Expected shape is (1, 50).'
        assert A1.shape == (1, 300), f'[Error] shape of A1 is {A1.shape}. Expected shape is (1, 300).'
        assert X.shape == (1, 769), f'[Error] shape of X is {X.shape}. Expected shape is (1, 769).'

        dEdA2 = Y_pred - Y_truth
        assert dEdA2.shape == (1, 50), f'[Error] shape of dEdA2 is {dEdA2.shape}. Expected shape is (1, 50).'

        dZ2_local = np.multiply(1 - A2, A2)
        dEdZ2 = np.multiply(dZ2_local, dEdA2)
        assert dEdZ2.shape == (1, 50), f'[Error] shape of dEdZ2 is {dEdZ2.shape}. Expected shape is (1, 50).'

        dEdW2 = np.outer(A1, dEdZ2)
        assert dEdW2.shape == (300, 50), f'[Error] shape of dEdW2 is {dEdW2.shape}. Expected shape is (300, 50).'

        dEdA1 = np.dot(dEdZ2, W2.T)
        assert dEdA1.shape == (1, 300), f'[Error] shape of dEdA1 is {dEdA1.shape}. Expected shape is (1, 300).'

        dZ1_local = np.multiply(1 - A1, A1)
        dEdZ1 = np.multiply(dZ1_local, dEdA1)
        assert dEdZ1.shape == (1, 300), f'[Error] shape of dEdZ1 is {dEdZ1.shape}. Expected shape is (1, 300).'

        dEdW1 = np.outer(X, dEdZ1)
        assert dEdW1.shape == (769, 300), f'[Error] shape of dEdW1 is {dEdW1.shape}. Expected shape is (769, 300).'

        dEdX = np.dot(dEdZ1, W1.T)
        assert dEdX.shape == (1, 769), f'[Error] shape of dEdX is {dEdX.shape}. Expected shape is (1, 769).'

        return dEdW1, dEdW2


    def cross_entropy(self, Y_pred, Y_truth):
        '''
        [Input]
          1. <ndarray> Y_pred: Its shape should be (1, 50).
          2. <ndarray> Y_truth: Its shape should be (1, 50).

        [Output]
          2. <ndarray> Error
        '''
        assert Y_truth.shape == (1, 50), f'[Error] shape of truth Y is {Y_truth.shape}. Expected shape is (1, 50).'
        assert Y_pred.shape == (1, 50), f'[Error] shape of predict Y is {S.shape}. Expected shape is (1, 50).'
        Error = (-1 * Y_truth * np.log(Y_pred)).sum()
        return Error


    def initialize_weights(self):
        '''
        [Input] None

        [Output]
          1. <ndarray> W1: Its shape should be (769, 300)
          2. <ndarray> W2: Its shape should be (300, 50)
        '''

        np.random.seed(0)
        W1 = np.random.uniform(low=-0.01, high=0.01, size=(769, 300))
        W2 = np.random.uniform(low=-0.01, high=0.01, size=(300, 50))
        assert W1.shape == (769, 300), f'[Error] shape of W1 is {W1.shape}. Expected shape is (769, 300).'
        assert W2.shape == (300, 50), f'[Error] shape of W2 is {W2.shape}. Expected shape is (300, 50).'
        return W1, W2


    def update_weights(self, dEdW1, dEdW2, W1, W2, lr):
        '''
        [Input]
          1. <ndarray> dEdW1
          2. <ndarray> dEdW2
          3. <ndarray> W1
          4. <ndarray> W2
          5. <float> lr

        [Output]
          1. <ndarray> W1
          2. <ndarray> W2
        '''
        W1 = W1 - lr * dEdW1
        W2 = W2 - lr * dEdW2
        return W1, W2


    def top_accuracy(self, Dataset, Label, W1, W2, Scale, Name):
        '''
        [Input
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
            # 1. Grab the i-th data
            X = np.array(Dataset[i:i + 1]).reshape(1, 769) / Scale
            Y = int(Label[i:i + 1][0])
            assert X.shape == (1, 769), f'[Error] shape of X is {X.shape}. Expected shape is (1, 769).'

            # 2. Predict the label by using Softmax.
            Y_pred, A1, A2 = self.forward_pass(X, W1, W2)
            assert Y_pred.shape == (1, 50), f'[Error] shape of predict Y is {Y_pred.shape}. Expected shape is (1, 50).'

            # 3. get top 5 predictions.
            top_1, top_2, top_3, top_4, top_5 = self.get_top_5_predictions(Y_pred)

            # 4. Check if the label is the top 1 prediction.
            if Y == top_1: num_top1_pred = num_top1_pred + 1

            # 5. Check if the label is in the top 5 predictions
            if Y in [top_1, top_2, top_3, top_4, top_5]: num_top5_pred = num_top5_pred + 1

        top1_accuracy = round(num_top1_pred / len_dataset * 100, 2)
        top5_accuracy = round(num_top5_pred / len_dataset * 100, 2)
        print(f'[Result of {Name}] The top-1 accuracy is {top1_accuracy} %')
        print(f'[Result of {Name}] The top-5 accuracy is {top5_accuracy} %')
        return top1_accuracy, top5_accuracy


    def get_top_5_predictions(self, Y_pred):
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


    def sigmoid(self, Z):
        '''
        [Input Vars]
          1. <ndarray> Z

        [Output Vars]
          1. <ndarray> A
        '''
        A = 1 / (1 + np.exp(-Z))
        return A


    def softmax(self, A):
        '''
        [Input Vars]
          1. <ndarray> A

        [Output Vars]
          1. <ndarray> Y_pred
        '''
        Y_pred = np.exp(A - np.max(A)) / np.sum(np.exp(A - np.max(A)))
        return Y_pred