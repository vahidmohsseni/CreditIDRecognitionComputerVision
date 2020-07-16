import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import cv2
import numpy as np


def train():
    (train_x, train_y) , (test_x, test_y) = mnist.load_data()
    #train_x = train_x.astype('float32') / 255
    #test_x = test_x.astype('float32') / 255print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    train_x = train_x.reshape(60000,784)
    test_x = test_x.reshape(10000,784)
    train_y = keras.utils.to_categorical(train_y,10)
    test_y = keras.utils.to_categorical(test_y,10)

    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_shape=(784,)))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))
    model.compile(optimizer=SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_x, train_y, batch_size=32, epochs=10, verbose=1)
    accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)
    print("Accuracy: ", accuracy[1])
    model.save("data/mnist_nn.h5")


def load_model():
    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_shape=(784,)))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))
    model.compile(optimizer=SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    model.load_weights("data/mnist_nn.h5")
    return model


def predict_from_src(pic):
    img = cv2.imread(pic, 0)
    # print(img.shape)
    # img = np.reshape(img, (3, 28,28,1))
    img = np.pad(img, ((5, 5), (5, 5)), 'constant', constant_values=255)
    plt.imshow(img)
    plt.show()
    img = img.reshape((1, 784))
    model = load_model()

    img_c = model.predict_classes(img)

    pred = img_c[0]
    return pred


def predict(img):
    # plt.imshow(img)
    # plt.show()
    img = np.pad(img, ((5, 5), (7, 8)), 'constant', constant_values=0)
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()
    img = img.reshape((1, 784))
    model = load_model()

    img_c = model.predict_classes(img)

    pred = img_c[0]
    return pred


if __name__ == "__main__":
    print(predict_from_src("5.png"))