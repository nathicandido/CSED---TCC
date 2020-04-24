import operator
from typing import List

from lucas_kanade.helpers.lkconstants import LKConstants
from lucas_kanade.helpers.real_car import RealCar
from utils.img_car import ImgCar
from utils.distance_calculator import DistanceCalculator


class TrackingController:

    def __init__(self):
        self.car_list: List[RealCar] = list()

    def receiver(self, img_car_list: List[ImgCar]):
        for new_image_car in img_car_list:
            best_candidates = self.get_ordered_list(new_image_car.get_position())
            new_real_car = RealCar(new_image_car)
            for candidate in best_candidates:
                if self.is_to_track(new_real_car, self.car_list[candidate[0]]):
                    self.car_list[candidate[0]].set_new_position(new_real_car.get_position())
                    self.car_list[candidate[0]].set_new_features(new_real_car.get_features())
                    break
            else:
                self.car_list.append(new_real_car)

    def get_ordered_list(self, target_position):
        list_by_distance = list()
        for index, car in enumerate(self.car_list):
            distance = target_position.get_distance(car.get_position())
            if distance < LKConstants.SEARCH_THRESHOLD_ON_THE_SURFACE:
                list_by_distance.append([index, distance])
        list_by_distance.sort(key=operator.itemgetter(1))
        return list_by_distance

    @staticmethod
    def is_to_track(a_car, b_car):
        if DistanceCalculator.n_dim_euclidean_distance(a_car.get_features(), b_car.get_features()) \
                < LKConstants.DISTANCE_TO_TRACK_THRESHOLD:
            return True
        return False
