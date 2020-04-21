import time
from typing import List

import numpy

from src.lucas_kanade.utils.point import Point
from src.yolo.utils.img_car import ImgCar


class RealCar:
    def __init__(self, img_car: ImgCar):
        self.ID = time.time()
        self.positions: List[Point] = list()
        self.positions.append(img_car.get_position())
        self.features = self.get_features_from_image(img_car.get_image())

    def get_position(self):
        return self.positions[-1]

    def get_distance(self, real_car):
        return self.get_distance(real_car.get_position())

    @staticmethod
    def get_features_from_image(image):
        avg_color_per_row = numpy.average(image, axis=0)
        return numpy.average(avg_color_per_row, axis=0)

    def set_new_position(self, new_position):
        self.positions.append(new_position)

    def set_new_features(self, new_features):
        self.features = new_features

    def get_features(self):
        return self.features
