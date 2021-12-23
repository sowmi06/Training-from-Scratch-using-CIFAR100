from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D, Flatten, Conv2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import cv2
import numpy as np
import time as t


def Cifar100_dataset():
    # loading CIFAR 100 from Keras
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # preprocessing the dataset
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 225.0
    x_test = x_test / 225.0
    # calling resize function on x_train and x_test
    x_train = resize(x_train)
    x_test = resize(x_test)
    # one hot code for y_train and y_test
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def training_from_scratch():
    # load the dataset
    x_train, y_train, x_test, y_test = Cifar100_dataset()
    # creating a sequential model for deep convolution neural network
    dcnn = Sequential()
    # adding convolution layer with input size 64x64x3, kernal size 3x3 and  stride 1,1
    dcnn.add(Conv2D(16, kernel_size=(3, 3), input_shape=(64, 64, 3), strides=(1, 1), padding='same', activation='relu'))
    dcnn.add(MaxPooling2D())
    dcnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    dcnn.add(MaxPooling2D())
    dcnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    dcnn.add(MaxPooling2D())
    dcnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    dcnn.add(MaxPooling2D())
    dcnn.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    dcnn.add(MaxPooling2D())
    dcnn.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    dcnn.add(MaxPooling2D())
    dcnn.add(GlobalAveragePooling2D())
    # flatten the features to fit in the fully connetced layer
    dcnn.add(Flatten())
    dcnn.add(Dropout(0.2))
    # adding fully connected layers to the DCNN
    dcnn.add(Dense(units=1000, activation='relu'))
    dcnn.add(Dense(units=500, activation='relu'))
    dcnn.add(Dense(units=100, activation='softmax'))

    # setting different learning rates for different convolution layers
    lr = ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.5)
    optimizer = Adam(learning_rate=lr)
    # compiling the DCNN
    dcnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # start time
    start_time = t.time()
    # trainig the DCNN
    dcnn.fit(x_train, y_train, epochs=300, steps_per_epoch=x_train.shape[0]/128, workers=8, validation_data=(x_test, y_test))
    # end time
    end_time = t.time()
    total_time = end_time - start_time
    print('Training Time:', total_time)


# method to resize image size
def resize(image):
    temp = []
    for i in range(len(image)):
        temp.append(cv2.resize(image[i], (64, 64)))
    return np.asarray(temp)


def main():
    training_from_scratch()


if __name__ == "__main__":
    main()


