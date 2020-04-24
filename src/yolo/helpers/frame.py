from typing import List

from yolo.helpers.yoloconstants import YOLOConstants
from utils.img_car import ImgCar


class Frame:

    def __init__(self, cars: List[ImgCar]):
        self.cars = self.delete_duplicated_cars(cars)

    def delete_duplicated_cars(self, cars: List[ImgCar]):
        not_duplicated_list: List[ImgCar] = list()
        for car in cars:
            if not self.is_the_same_car(car.get_position(), not_duplicated_list):
                not_duplicated_list.append(car)
        return not_duplicated_list

    @staticmethod
    def is_the_same_car(point, cars: List[ImgCar]):
        for car in cars:
            if point.get_distance(car.get_position()) < YOLOConstants.DISTANCE_TO_BE_THE_SAME_CAR:
                return True
        else:
            return False
