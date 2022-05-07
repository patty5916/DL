from keras import models, layers

def LeNet5():
    model = models.Sequential()
    model.add(layers.Conv2D(6, 5, input_shape=(128, 128, 3)))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(16, 5))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(120))
    model.add(layers.Dense(84))
    model.add(layers.Dense(50, activation='softmax'))
    model.summary()

    return model
