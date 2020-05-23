""" main.py """
from pathlib import Path
from argparse import ArgumentParser, ArgumentError
from itertools import starmap
from matplotlib import pyplot as plt
from numpy import savez, array
from tqdm import tqdm
import time
from os import rename

from lucas_kanade.tracking_controller import TrackingController
from constants.general_parameters import GeneralParameters
from yolo.helpers.camera import Camera
from yolo.yolo_controller import YoloController
from fft.fourier_controller import FourierController


class Main:
    ANGLE_LIST = [0, 90, 180, 270]
    PARSED_FILES_PATH = Path.joinpath(Path.cwd(), '..', 'dataset', 'parsed')

    @classmethod
    def filter_irrevelevant_ts(cls, tracker: TrackingController, maneuver_dataset_index):
        for i, car in enumerate(tracker.car_list):
            if len(car.positions) < GeneralParameters.INSUFFICIENT_TIME_SERIES_LENGTH:
                try:
                    rename(
                        str(Path.joinpath(GeneralParameters.SAVED_IMAGES_FOLDER,
                                          f'idx_{maneuver_dataset_index}-{str(car.ID)}')),
                        str(Path.joinpath(GeneralParameters.SAVED_IMAGES_FOLDER,
                                          f'idx_{maneuver_dataset_index}-{str(car.ID)}_DELETED'))
                    )

                except FileNotFoundError:
                    pass
                tracker.car_list.pop(i)

        return tracker

    @classmethod
    def run(cls, video_path_list, parse=False, maneuver='', maneuver_dataset_index=0, plot=False, debug=False, dump_buffer=False):

        camera_list = list(
            starmap(
                lambda angle, path: Camera(
                    str(Path.joinpath(Path.cwd(), *(path.split('/')))),
                    angle
                ),
                zip(cls.ANGLE_LIST, video_path_list)
            )
        )

        min_video_length = min(list(map(lambda c: c.frame_count, camera_list)))

        received_ts = list()
        yolo = YoloController(cameras=camera_list, debug=debug)
        lucas_kanade = TrackingController(debug=debug, dump_buffer=dump_buffer, maneuver_dataset_index=maneuver_dataset_index)

        if not debug:
            for _ in tqdm(range(0, min_video_length, GeneralParameters.NUMBER_OF_FRAMES)):
                cars_list = yolo.get_cameras_images()
                received_ts.extend(lucas_kanade.receiver(cars_list))

        else:
            for _ in range(0, min_video_length, GeneralParameters.NUMBER_OF_FRAMES):
                cars_list = yolo.get_cameras_images()
                received_ts.extend(lucas_kanade.receiver(cars_list))

        lucas_kanade = cls.filter_irrevelevant_ts(lucas_kanade, maneuver_dataset_index)

        abstract_vehicle_list = list()

        for car in lucas_kanade.car_list:
            abstract_vehicle_list.append(
                FourierController.build_abstract_vehicle(
                    car.ID,
                    list(map(lambda c: c.x, car.positions)),
                    list(map(lambda c: c.y, car.positions))
                )
            )
        for car in received_ts:
            abstract_vehicle_list.append(
                FourierController.build_abstract_vehicle(
                    car.ID,
                    list(map(lambda c: c.x, car.positions)),
                    list(map(lambda c: c.y, car.positions))
                )
            )

        if plot:
            while True:
                plot_index = input('Enter vehicle index to be plotted (Q + Enter to exit): ')

                if plot_index.upper() == 'Q':
                    break

                try:
                    plt.plot(list(map(lambda p: p.x, lucas_kanade.car_list[int(plot_index)].positions)), label='X')
                    plt.plot(list(map(lambda p: p.y, lucas_kanade.car_list[int(plot_index)].positions)), label='Y')
                    plt.legend(loc='best')
                    plt.show()

                except TypeError:
                    print('Input value must be a integer or Q to exit')

                except IndexError:
                    print('Index out of bounds')

        if parse:
            current_parse_folder = Path.joinpath(cls.PARSED_FILES_PATH, str(time.time()).replace('.', '-'))
            current_parse_folder.mkdir(parents=True, exist_ok=True)
            for car in abstract_vehicle_list:
                savez(
                    f'{current_parse_folder}/'
                    f'{maneuver.upper()}_{str(car.vehicle_id).replace(".", "_")}-idx_{maneuver_dataset_index}.npz',
                    label=array([maneuver.upper()]),
                    x_sig=car.signal.x_sig,
                    y_sig=car.signal.y_sig
                )


if __name__ == '__main__':
    parser = ArgumentParser()

    optional_group = parser.add_argument_group('Optional Arguments')
    required_group = parser.add_argument_group('Required Arguments')

    optional_group.add_argument('--parse',
                                help='Boolean argument used to determine '
                                     'whether time series generated by the '
                                     'system should be stored for further training. '
                                     'Must inform --maneuver',
                                action='store_true')

    optional_group.add_argument('--maneuver',
                                help='Maneuver name so the parsed files may be saved properly')

    optional_group.add_argument('--plot',
                                help='Boolean argument used for plotting a time series. Must inform --plot_index',
                                action='store_true')

    optional_group.add_argument('--debug',
                                help='Visualize a more thorough execution',
                                action='store_true')

    optional_group.add_argument('--dump_buffer',
                                help='Save images from frame buffer during execution',
                                action='store_true')

    required_group.add_argument('--video_path_list',
                                help='Set of paths to access the videos for processing, '
                                     'i. e.: "path/to/file/one; path/to/file/2", '
                                     '4 files are required',
                                required=True)

    args = parser.parse_args()

    if all([args.parse, args.maneuver]):

        path_list = args.video_path_list.split(';')
        maneuver_dataset_index = path_list[0].split('/')[-2].split('_')[-1]
        if len(path_list) != 4:
            raise ArgumentError('--video_path_list must contain 4 paths')

        Main.run(
            path_list,
            parse=args.parse,
            maneuver=args.maneuver,
            maneuver_dataset_index=maneuver_dataset_index,
            plot=args.plot,
            debug=args.debug,
            dump_buffer=args.dump_buffer
        )

    else:
        raise ArgumentError('if --parse is active --maneuver is required')
