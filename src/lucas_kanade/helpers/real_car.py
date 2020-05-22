import time
from typing import List

import numpy

from logger.log import Log
from utils.point import Point
from utils.img_car import ImgCar
from constants.general_parameters import GeneralParameters


class RealCar:
    def __init__(self, img_car: ImgCar):
        self.TAG = 'RealCar'
        self.log = Log()
        self.ID = time.time()
        self.tracking_counter = 10
        self.was_detected = False
        self.positions: List[Point] = list()
        self.positions.append(img_car.get_position())
        self.features = self.get_features_from_image(img_car.get_image())

    def get_position(self):
        return self.positions[-1]

    def get_distance(self, real_car):
        return self.positions[-1].get_distance(real_car.get_position())

    @staticmethod
    def get_features_from_image(image):
        avg_color_per_row = numpy.average(image, axis=0)
        gray_scale = numpy.average(avg_color_per_row, axis=0)
        return gray_scale[0] * 0.11 + gray_scale[1] * 0.59 + gray_scale[2] * 0.3

    def set_new_position(self, new_position):
        self.positions.append(new_position)
        self.tracking_counter = GeneralParameters.LK_CAR_DETECTION_COUNTER
        self.was_detected = True

    def is_tracking(self):
        if not self.was_detected:
            self.tracking_counter -= 1
        self.was_detected = False
        return self.tracking_counter > 0

    def set_new_features(self, new_features):
        self.features = new_features

    def get_features(self):
        return self.features
