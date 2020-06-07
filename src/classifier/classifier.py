import os

from keras import backend
from keras.layers import Dense, Flatten, LeakyReLU
from keras.models import Sequential
from tensorflow import ConfigProto, Session
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle as pkl
from pprint import pprint as pp


class Classifier:
    LABELS = {"['SWERVING']": np.array([1, 0, 0, 0]),
              "['ULTRAPASSA']": np.array([0, 1, 0, 0]),
              "['FALSAULTRAPASSA']": np.array([0, 0, 1, 0]),
              "['MUDAFAIXA']": np.array([0, 0, 0, 1])}

    ACC_PER_CLASS = list()
    PPV_PER_CLASS = list()
    TPR_PER_CLASS = list()

    AVG_ACC = list()

    def __init__(self):

        config = ConfigProto(device_count=dict(GPU=1, CPU=2))
        sess = Session(config=config)
        backend.set_session(sess)
        self.run()
        backend.clear_session()

    @classmethod
    def get_mean(cls, metrics):
        return sum(metrics) / len(metrics)

    @classmethod
    def dataset_split_metrics(cls, train, test):
        train_test_split_list = [[0, 0], [0, 0], [0, 0], [0, 0]]

        for etr in train:
            train_test_split_list[int(np.argmax(etr))][0] += 1

        for etst in test:
            train_test_split_list[int(np.argmax(etst))][1] += 1

        return train_test_split_list

    @classmethod
    def extract_metrics(cls, predictions, expected):
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for pred, exp in zip(predictions, expected):
            if np.argmax(pred) == np.argmax(exp):
                tp += 1
                tn += 3
            else:
                fp += 1
                tn += 2
                fn += 1

        print(f'TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}')
        return tp, fp, tn, fn

    @classmethod
    def split_classes(cls, predictions, expected):
        splitted_prediction_list = [[], [], [], []]
        splitted_expected_list = [[], [], [], []]

        for pred, exp in zip(predictions, expected):
            splitted_prediction_list[int(np.argmax(exp))].append(pred)
            splitted_expected_list[int(np.argmax(exp))].append(exp)

        return splitted_prediction_list, splitted_expected_list

    @classmethod
    def metrics(cls, predictions, expected):
        splitted_prediction, splitted_expected = cls.split_classes(predictions, expected)
        metrics_matrix = list()
        for class_pred, class_exp in zip(splitted_prediction, splitted_expected):
            metrics_matrix.append(cls.extract_metrics(class_pred, class_exp))

        acc_tmp = list()
        ppv_tmp = list()
        tpr_tmp = list()
        for metrics_per_class in metrics_matrix:
            tp, fp, tn, fn = metrics_per_class
            acc_tmp.append(cls.accuracy(tp=tp, fp=fp, tn=tn, fn=fn))
            ppv_tmp.append(cls.precision(tp=tp, fp=fp))
            tpr_tmp.append(cls.recall(tp=tp, fn=fn))

        cls.ACC_PER_CLASS.append(acc_tmp)
        cls.PPV_PER_CLASS.append(ppv_tmp)
        cls.TPR_PER_CLASS.append(tpr_tmp)

        cls.AVG_ACC.append(cls.get_mean(acc_tmp))

    @classmethod
    def accuracy(cls, tp, fp, tn, fn):
        return (tp + tn) / (tp + tn + fp + fn)

    @classmethod
    def precision(cls, tp, fp):
        return tp / (tp + fp)

    @classmethod
    def recall(cls, tp, fn):
        return tp / (tp + fn)

    @classmethod
    def run(cls):

        original_train_dataset, original_train_labels = cls.get_dataset()

        # Will return a list of sliced dataset, to use cross validation
        train_dataset_list, train_labels_list, test_dataset_list, test_labels_list = \
            cls.split_dataset_with_cross_validation(original_train_dataset, original_train_labels)

        tr_tst_matrix = list()
        for train_dataset, train_labels, test_dataset, test_labels in zip(train_dataset_list, train_labels_list,
                                                                          test_dataset_list, test_labels_list):
            train_dataset = np.array(train_dataset)

            train_labels = np.array(list(map(lambda each: cls.LABELS[each], train_labels)))

            classifier = cls.create_classifier()

            classifier.fit(train_dataset, train_labels, batch_size=10, epochs=100)

            # To validate prediction_list with test_labels.
            # So, will be possible to confirm the accuracy for this sliced list

            test_dataset = np.array(test_dataset)
            test_labels = np.array(list(map(lambda each: cls.LABELS[each], test_labels)))

            prediction_list = classifier.predict(test_dataset)

            cls.metrics(prediction_list, test_labels)

            tr_tst_matrix.append(cls.dataset_split_metrics(train_labels, test_labels))

        print('AVG ACCURACY')
        for avg_acc in cls.AVG_ACC:
            print(avg_acc)

        print('ACC PER CLASS')
        for acc in cls.ACC_PER_CLASS:
            pp(acc)

        print('PPV PER CLASS')
        for ppv in cls.PPV_PER_CLASS:
            pp(ppv)

        print('TPR PER CLASS')
        for tpr in cls.TPR_PER_CLASS:
            pp(tpr)

        print('TRAIN TEST SPLIT')
        for trtst in tr_tst_matrix:
            print(trtst)

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
        with open(os.path.join(os.getcwd(), '..', '..', 'dataset', 'ready_for_training.pkl'), 'rb') as f:
            data = pkl.load(f)
            train_dataset = np.array([[index['x_sig'], index['y_sig']] for index in data])
            train_label = np.array([np.array2string(index['label']) for index in data])
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
            train_dataset, train_labels, test_dataset, test_labels = cls.split_dataset(train_index, test_index, dataset,
                                                                                       labels)
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
