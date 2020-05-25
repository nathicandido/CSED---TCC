from keras import backend
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.generic_utils import get_custom_objects
from tensorflow import ConfigProto, Session, function
import keras
import numpy as np


class Classifier:

    def __init__(self):
        @function
        def leaky_relu(x, alpha=0.01):
            print(f'(Func = {x.op}')
            if x.op < 0:
                x.op = x.op * alpha
                return x
            return x

        get_custom_objects().update(dict(leaky_relu=Activation(leaky_relu)))

        config = ConfigProto(device_count=dict(GPU=1, CPU=2))
        sess = Session(config=config)
        self.run()
        backend.set_session(sess)
        backend.clear_session()

    @classmethod
    def run(cls):

        # full_array = np.array([np.array([x, y]) for x, y in zip(arr_x, arr_y)])
        # result = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
        # full_test = np.array([np.array([x, y]) for x, y in zip(test_x, test_y)])

        # Creating 4 Layers (2*100, 2*52, 2*26, 4)
        classifier = Sequential()
        classifier.add(LeakyReLU(alpha=0.3, input_shape=(2, 100)))
        classifier.add(Dense(units=52, activation="relu"))
        classifier.add(Dense(units=26, activation="relu"))
        classifier.add(Flatten())

        classifier.add(Dense(units=4, activation="softmax"))
        classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # classifier.fit(TRAINNING, RESULTS, batch_size=1, epochs=10)

        # classifier.predict(TEST))

if __name__ == '__main__':
    Classifier()