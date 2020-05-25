import os

from keras import backend
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.models import Sequential
from keras.utils.generic_utils import get_custom_objects
from tensorflow import ConfigProto, Session, function
from sklearn.model_selection import StratifiedKFold
import keras
import numpy as np
import pickle as pkl


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

        classifier = cls.create_classifier()
        original_train_dataset, original_train_labels = cls.get_dataset()

        # Will return a list of sliced dataset, to use cross validation
        train_dataset_list, train_labels_list, test_dataset_list, test_labels_list = \
            cls.split_dataset_with_cross_validation(original_train_dataset, original_train_labels)

        for train_dataset, train_labels, test_dataset, test_labels in zip(train_dataset_list, train_labels_list, test_dataset_list, test_labels_list):
            classifier.fit(train_dataset, train_labels, batch_size=1, epochs=100)

            # To validate prediction_list with test_labels.
            # So, will be possible to confirm the accuracy for this sliced list
            prediction_list = classifier.predict(test_dataset)

        # will be necessary to confirm all predicted values, and will be possible to determine the accuracy for our model/classifier.


    @classmethod
    def create_classifier(self):
        # Creating 4 Layers (2*100, 2*52, 2*26, 4)
        classifier = Sequential()
        classifier.add(LeakyReLU(alpha=0.3, input_shape=(2, 100)))
        classifier.add(Dense(units=52, activation="relu"))
        classifier.add(Dense(units=26, activation="relu"))
        classifier.add(Flatten())
        classifier.add(Dense(units=4, activation="softmax"))
        classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return classifier

    @classmethod
    def get_dataset(cls):
        with open(os.path.join(os.getcwd(), '..', 'dataset', 'ready_for_training.pkl'), 'rb') as f:
            data = pkl.load(f)
            labels = {"SWERVING": [1, 0, 0, 0],
                      "ULTRAPASSA": [0, 1, 0, 0],
                      "FALSAULTRAPASSA": [0, 0, 1, 0],
                      "MUDAFAIXA": [0, 0, 0, 1]}
            train_dataset = np.array([np.array([index['x_sig'], index['y_sig']]) for index in data])
            train_label = np.array([(labels[index['label']]) for index in data])
            return train_dataset, train_label

    @classmethod
    def split_dataset_with_cross_validation(cls, dataset, labels, k=5):
        splitter = StratifiedKFold(n_splits=k)
        cross_validation = splitter.split(dataset, labels)
        crossed_train_dataset_list = list()
        crossed_train_labels_list = list()
        crossed_test_dataset_list = list()
        crossed_test_labels_list = list()
        for train_index, test_index in cross_validation:
            train_dataset, train_labels, test_dataset, test_labels = cls.split_dataset(train_index, test_index, dataset, labels)
            crossed_train_dataset_list.append(train_dataset)
            crossed_train_labels_list.append(train_labels)
            crossed_test_dataset_list.append(test_dataset)
            crossed_test_labels_list.append(test_labels)
        return crossed_train_dataset_list, crossed_train_labels_list, crossed_test_dataset_list, crossed_test_labels_list

    @classmethod
    def split_dataset(cls, train_index, test_index, dataset, labels):
        train_dataset, train_labels = [(dataset[index], labels[index]) for index in train_index]
        test_dataset, test_labels = [(dataset[index], labels[index]) for index in test_index]
        return train_dataset, train_labels, test_dataset, test_labels


if __name__ == '__main__':
    Classifier()