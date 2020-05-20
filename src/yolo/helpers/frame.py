from typing import List

from utils.point import Point
from yolo.helpers.yoloconstants import YOLOConstants
from utils.img_car import ImgCar


class Frame:

    def __init__(self, cars: List[ImgCar]):
        self.cars = self.delete_duplicated_cars(cars)

    def delete_duplicated_cars(self, cars: List[ImgCar]):
        not_duplicated_list: List[ImgCar] = list()
        for car in cars:
            duplicated_car = self.is_the_same_car(car.get_b_box_center(), not_duplicated_list)
            if duplicated_car:
                reference = Point(0, 0)
                if reference.get_distance(car.get_position()) < reference.get_distance(duplicated_car.get_position()):
                    not_duplicated_list.remove(duplicated_car)
                    not_duplicated_list.append(car)
                    # print(f'Size {len(not_duplicated_list)}')
            else:
                not_duplicated_list.append(car)
        return not_duplicated_list

    @staticmethod
    def is_the_same_car(point, cars: List[ImgCar]):
        for car in cars:
            if point.get_distance(car.get_b_box_center()) < YOLOConstants.DISTANCE_TO_BE_THE_SAME_CAR:
                # print(f'Deleta - {point.get_distance(car.get_b_box_center())}')
                # print(f'Centro A = {point},  Centro B = {car.get_b_box_center()}')
                return car
        else:
            return None
