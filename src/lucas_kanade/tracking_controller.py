import operator
from pathlib import Path
from typing import List
from os import rename

import cv2

from logger.log import Log
from constants.general_parameters import GeneralParameters
from lucas_kanade.helpers.real_car import RealCar
from utils.img_car import ImgCar
import time


class TrackingController:

    def __init__(self, debug=False, dump_buffer=False, maneuver_dataset_index=0):
        self.TAG = 'LK'
        self.debug = debug
        self.dump_buffer = dump_buffer
        self.log = Log()
        self.car_list: List[RealCar] = list()
        if self.dump_buffer:
            self.maneuver_dataset_index = maneuver_dataset_index
        try:
            Path.mkdir(GeneralParameters.SAVED_IMAGES_FOLDER)

        except FileExistsError:
            pass

    def receiver(self, img_car_list: List[ImgCar]):
        if self.debug:
            self.log.i(self.TAG, f'Number of cars is {len(img_car_list)}')

        for new_image_car in img_car_list:
            best_candidates = self.get_ordered_list(new_image_car.get_position())
            new_real_car = RealCar(new_image_car)
            for candidate in best_candidates:
                if self.is_to_track(new_real_car, self.car_list[candidate[0]]):

                    self.car_list[candidate[0]].set_new_position(new_real_car.get_position())
                    self.car_list[candidate[0]].set_new_features(new_real_car.get_features())

                    if self.debug:
                        self.log.d(self.TAG, f'is the same car, ID {self.car_list[candidate[0]].ID}')

                    if self.dump_buffer:
                        cv2.imwrite(str(Path.joinpath(GeneralParameters.SAVED_IMAGES_FOLDER, f'{str(self.car_list[candidate[0]].ID)}-idx_{self.maneuver_dataset_index}',
                                                      f'{time.time()}.jpg')), new_image_car.get_image())
                    break
            else:
                if self.debug:
                    self.log.d(self.TAG, f'is a different car, new car ID is: {new_real_car.ID}')

                if self.dump_buffer:
                    if not Path.joinpath(GeneralParameters.SAVED_IMAGES_FOLDER, str(new_real_car.ID)).is_dir():
                        Path.mkdir(Path.joinpath(GeneralParameters.SAVED_IMAGES_FOLDER, f'{str(new_real_car.ID)}-idx_{self.maneuver_dataset_index}'))
                        cv2.imwrite(str(Path.joinpath(GeneralParameters.SAVED_IMAGES_FOLDER, f'{str(new_real_car.ID)}-idx_{self.maneuver_dataset_index}',
                                                      f'{time.time()}.jpg')), new_image_car.get_image())
                self.car_list.append(new_real_car)

        self.check_tracking_cars()

    def get_ordered_list(self, target_position):
        list_by_distance = list()
        for index, car in enumerate(self.car_list):
            distance = target_position.get_distance(car.get_position())
            if self.debug:
                self.log.i(self.TAG, f'Plane Distance: {distance}')
            if distance < GeneralParameters.LK_SEARCH_THRESHOLD_ON_THE_SURFACE:
                list_by_distance.append([index, distance])
        list_by_distance.sort(key=operator.itemgetter(1))
        return list_by_distance

    def check_tracking_cars(self):
        for index, car in enumerate(self.car_list):
            if not car.is_tracking():
                if self.debug:
                    self.log.w(self.TAG, f'Car {car.ID} was lost, deleting time series')
                try:
                    rename(
                        str(Path.joinpath(GeneralParameters.SAVED_IMAGES_FOLDER, f'{str(car.ID)}-idx_{self.maneuver_dataset_index}')),
                        str(Path.joinpath(GeneralParameters.SAVED_IMAGES_FOLDER, f'{str(car.ID)}-idx_{self.maneuver_dataset_index}_DELETED'))
                    )

                except FileNotFoundError:
                    pass
                self.car_list.pop(index)

    def is_to_track(self, a_car, b_car):
        if abs(a_car.get_features() - b_car.get_features()) < GeneralParameters.LK_DISTANCE_TO_TRACK_THRESHOLD:
            return True
        if self.debug:
            self.log.e(self.TAG,
                       f'Grayscale Distance: {abs(a_car.get_features() - b_car.get_features())}')
        return False
