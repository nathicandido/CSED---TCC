import numpy as np
import os
import pickle as pkl
import itertools as it
from pathlib import Path
from fft.fourier_controller import FourierController
from json_dataset_builder import JSONDatasetBuilder


class ClassedPattern:
    def __init__(self, x_sig, y_sig, label):
        self.x_sig = x_sig
        self.y_sig = y_sig
        self.label = label

    def __repr__(self):
        return f'<ClassedPattern: {self.label}>'


class SynthetizedClassedPattern(ClassedPattern):
    def __init__(self, x_sig, y_sig, label):
        super().__init__(x_sig, y_sig, label)

    def __repr__(self):
        return f'<SynthetizedClassedPattern: {self.label}>'


class GaussianNoiseSynthetizer:
    PATH_TO_PARSED = str(Path.joinpath(Path.cwd(), '..', 'dataset', 'parsed'))
    TARGET_TO_FULL_DATASET = str(Path.joinpath(Path.cwd(), '..', 'dataset', 'ready_for_training'))

    CLASS_ARRAY_X_KEY = 'x_sig'
    CLASS_ARRAY_Y_KEY = 'y_sig'
    CLASS_ARRAY_LABEL_KEY = 'label'

    PICKLE_OPENING_MODE = 'wb'

    def __init__(self):
        self.arrays = [
            np.load(f) for f in self.absolute_file_paths(self.PATH_TO_PARSED)
        ]

        x_sig_group = map(lambda a: a[self.CLASS_ARRAY_X_KEY], self.arrays)
        y_sig_group = map(lambda a: a[self.CLASS_ARRAY_Y_KEY], self.arrays)
        label_group = map(lambda a: a[self.CLASS_ARRAY_LABEL_KEY], self.arrays)

        self.classed_patterns, self.classed_patterns_bkp = it.tee(
            it.starmap(
                lambda x, y, l: ClassedPattern(x_sig=x, y_sig=y, label=l),
                zip(x_sig_group, y_sig_group, label_group)
            )
        )

    def serialize_dataset(self, array):
        with open(f'{self.TARGET_TO_FULL_DATASET}.pkl', self.PICKLE_OPENING_MODE) as pkl_in:
            pkl.dump(array, pkl_in)

    @staticmethod
    def absolute_file_paths(directory):
        for dirpath, _, filenames in os.walk(directory):
            for file in filenames:
                yield os.path.abspath(os.path.join(dirpath, file))

    @staticmethod
    def generate_n_gaussian_samples_from_unit(n, scale_x, scale_y, classed_pattern):

        data = list()

        for _ in range(n):
            x_noise = [x + gss for gss, x in zip(np.random.normal(0, scale_x, 100), classed_pattern.x_sig)]
            y_noise = [y + gss for gss, y in zip(np.random.normal(0, scale_y, 100), classed_pattern.y_sig)]

            data.append(SynthetizedClassedPattern(x_sig=x_noise, y_sig=y_noise, label=classed_pattern.label))

        return data

    def generate_gaussian_samples_from_classed_patterns(self, n=100, scale_x=1, scale_y=.5):
        synth_classed_patterns_from_gaussian = [
            self.generate_n_gaussian_samples_from_unit(n=n, scale_x=scale_x, scale_y=scale_y, classed_pattern=cp)
            for cp in self.classed_patterns
        ]

        for pattern in synth_classed_patterns_from_gaussian:
            for cp in pattern:
                cp.x_sig = FourierController.smoothen_and_interpolate(cp.x_sig)
                cp.y_sig = FourierController.smoothen_and_interpolate(cp.y_sig)

        return synth_classed_patterns_from_gaussian

    def run(self):

        dataset = list()

        patterns = self.generate_gaussian_samples_from_classed_patterns()

        dataset.extend(JSONDatasetBuilder.build_json_from_patterns(patterns=patterns))
        dataset.extend(JSONDatasetBuilder.build_json_from_patterns(patterns=self.classed_patterns_bkp))

        self.serialize_dataset(dataset)


if __name__ == '__main__':
    GaussianNoiseSynthetizer().run()
