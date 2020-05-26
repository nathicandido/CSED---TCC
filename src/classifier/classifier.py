import os

from keras import backend
from keras.layers import Dense, Flatten, LeakyReLU
from keras.models import Sequential
from tensorflow import ConfigProto, Session
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle as pkl
from time import time


class Classifier:

    LABELS = {"['SWERVING']": np.array([1, 0, 0, 0]),
              "['ULTRAPASSA']": np.array([0, 1, 0, 0]),
              "['FALSAULTRAPASSA']": np.array([0, 0, 1, 0]),
              "['MUDAFAIXA']": np.array([0, 0, 0, 1])}

    def __init__(self):

        config = ConfigProto(device_count=dict(GPU=1, CPU=2))
        sess = Session(config=config)
        backend.set_session(sess)
        self.run()
        backend.clear_session()

    @classmethod
    def run(cls):

        original_train_dataset, original_train_labels = cls.get_dataset()

        # Will return a list of sliced dataset, to use cross validation
        train_dataset_list, train_labels_list, test_dataset_list, test_labels_list = \
            cls.split_dataset_with_cross_validation(original_train_dataset, original_train_labels)

        for train_dataset, train_labels, test_dataset, test_labels in zip(train_dataset_list, train_labels_list, test_dataset_list, test_labels_list):
            train_dataset = np.array(train_dataset)
            train_labels = np.array(list(map(lambda each: cls.LABELS[each], train_labels)))

            classifier = cls.create_classifier()

            classifier.fit(train_dataset, train_labels, batch_size=10, epochs=100)

            # To validate prediction_list with test_labels.
            # So, will be possible to confirm the accuracy for this sliced list

            test_dataset = np.array(test_dataset)
            test_labels = np.array(list(map(lambda each: cls.LABELS[each], test_labels)))

            prediction_list = classifier.predict(test_dataset)

            t = str(time())
            np.savez(
                f'C:/Users/Tiagoo/PycharmProjects/CSED---TCC/dataset/results/result_{t.replace(".", "_")}.npz',
                prediction=prediction_list,
                expected=test_labels
            )

            acc = 0
            for pred, expected in zip(prediction_list, test_labels):
                if np.argmax(pred) == np.argmax(expected):
                    acc += 1

            print(acc / len(prediction_list))

        # will be necessary to confirm all predicted values,
        # and will be possible to determine the accuracy for our model/classifier.

    @classmethod
    def create_classifier(cls):
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
            train_dataset = np.array([[index['x_sig'], index['y_sig']] for index in data])
            train_label = np.array([np.array2string(index['label']) for index in data])
            return train_dataset, train_label

    @classmethod
    def split_dataset_with_cross_validation(cls, dataset, labels, k=3):
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
        train_dataset, train_labels = [
            [dataset[index] for index in train_index],
            [labels[index] for index in train_index]
        ]

        test_dataset, test_labels = [
            [dataset[index] for index in test_index],
            [labels[index] for index in test_index]
        ]

        return train_dataset, train_labels, test_dataset, test_labels


if __name__ == '__main__':
    Classifier()