from itertools import starmap
from math import sqrt


class DistanceCalculator:

    @classmethod
    def _fill_with_zeroes(cls, vector_1: list, vector_2: list):
        len_diff = len(vector_1) - len(vector_2)

        if len_diff < 0:
            vector_1.extend(abs(len_diff) * [0])
        elif len_diff > 0:
            vector_2.extend(abs(len_diff) * [0])

        return vector_1, vector_2

    @classmethod
    def n_dim_euclidean_distance(cls, point_1, point_2):
        point_1, point_2 = cls._fill_with_zeroes(point_1, point_2)

        return sqrt(sum(starmap(lambda d1, d2: (d2 - d1) ** 2, zip(point_1, point_2))))
