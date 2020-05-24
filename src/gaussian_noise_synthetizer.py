import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
from pathlib import Path
from fft.fourier_controller import FourierController
from json_dataset_builder import JSONDatasetBuilder


class GaussianNoiseSynthetizer:

    PATH_TO_PARSED = str(Path.joinpath(Path.cwd(), '..', 'dataset', 'parsed'))
    TARGET_TO_FULL_DATASET = str(Path.joinpath(Path.cwd(), '..', 'dataset', 'ready_for_training'))

    CLASS_ARRAY_X_KEY = 'x_sig'
    CLASS_ARRAY_Y_KEY = 'y_sig'
    CLASS_ARRAY_LABEL_KEY = 'label'

    PICKLE_OPENING_MODE = 'wb'
    LEGEND_PLOT_LOCATION = 'best'

    def __init__(self):
        self.arrays = [
            np.load(f) for f in self.absolute_file_paths(self.PATH_TO_PARSED)
        ]

        self.x_sig_group = list(map(lambda a: a[self.CLASS_ARRAY_X_KEY], self.arrays))
        self.y_sig_group = list(map(lambda a: a[self.CLASS_ARRAY_Y_KEY], self.arrays))
        self.label_group = map(lambda a: a[self.CLASS_ARRAY_LABEL_KEY], self.arrays)

        self.class_ = next(self.label_group)[0]

        x_pos_group = [pos for pos in zip(*self.x_sig_group)]
        y_pos_group = [pos for pos in zip(*self.y_sig_group)]

        self.consolidate = [(x, y) for x, y in zip(x_pos_group, y_pos_group)]

    def serialize_dataset(self, array):
        date = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
        with open(f'{self.TARGET_TO_FULL_DATASET}_{self.class_}_{date}.pkl', self.PICKLE_OPENING_MODE) as pkl_in:
            pkl.dump(array, pkl_in)

    @staticmethod
    def absolute_file_paths(directory):
        for dirpath, _, filenames in os.walk(directory):
            for file in filenames:
                yield os.path.abspath(os.path.join(dirpath, file))

    @staticmethod
    def generate_gaussian_noise(n, scale, signal_):
        gaussian_noise = [[p + n for p, n in zip(signal_, np.random.normal(0, scale, 100))] for _ in range(n)]
        return gaussian_noise

    def get_mean_array(self):
        mean_arr_x = list()
        mean_arr_y = list()

        for c in self.consolidate:
            mean_arr_x.append(sum(c[0]) / len(c[0]))
            mean_arr_y.append(sum(c[1]) / len(c[1]))

        return mean_arr_x, mean_arr_y

    def generate_gaussian_noise_from_mean(self, samples=100, plot=False):
        mean_x, mean_y = self.get_mean_array()

        gaussian_noise_x = self.generate_gaussian_noise(n=samples, scale=1, signal_=mean_x)
        gaussian_noise_y = self.generate_gaussian_noise(n=samples, scale=2, signal_=mean_y)

        if plot:
            for a in self.x_sig_group:
                plt.plot(a)

            for gx in gaussian_noise_x:
                plt.plot(FourierController.smoothen_and_interpolate(gx))

            plt.plot(mean_x, 'o', label='Mean X')
            plt.title(f'X with Gaussian Noise ({samples} samples)')
            plt.legend(loc=self.LEGEND_PLOT_LOCATION)
            plt.show()

            plt.clf()
            plt.cla()
            plt.close()

            for a in self.y_sig_group:
                plt.plot(a)

            for gy in gaussian_noise_y:
                plt.plot(FourierController.smoothen_and_interpolate(gy))

            plt.plot(mean_x, 'o', label='Mean y')
            plt.title(f'Y with Gaussian Noise ({samples} samples)')
            plt.legend(loc=self.LEGEND_PLOT_LOCATION)
            plt.show()

            plt.clf()
            plt.cla()
            plt.close()

    def generate_gaussian_noise_from_class_arrays(self, n_x=100, n_y=100, scale_x=1, scale_y=.5, plot=False):
        gaussian_x = [self.generate_gaussian_noise(n=n_x, scale=scale_x, signal_=x) for x in self.x_sig_group]
        gaussian_y = [self.generate_gaussian_noise(n=n_y, scale=scale_y, signal_=y) for y in self.y_sig_group]

        sm_gaussian_x = [[FourierController.smoothen_and_interpolate(g) for g in matrix] for matrix in gaussian_x]
        sm_gaussian_y = [[FourierController.smoothen_and_interpolate(g) for g in matrix] for matrix in gaussian_y]

        if plot:
            for index, (x_matrix, y_matrix) in enumerate(zip(sm_gaussian_x, sm_gaussian_y)):
                for x_signal, y_signal in zip(x_matrix, y_matrix):
                    plt.plot(x_signal)
                    plt.plot(y_signal)

                plt.title(index)
                plt.show()
                plt.cla()
                plt.clf()
                plt.close()

        return sm_gaussian_x, sm_gaussian_y

    def run(self):
        smg_x, smg_y = self.generate_gaussian_noise_from_class_arrays()
        data = JSONDatasetBuilder.build_json_for_training(smg_x, smg_y, self.class_)
        self.serialize_dataset(data)


if __name__ == '__main__':
    GaussianNoiseSynthetizer().run()
