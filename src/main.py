""" main.py """
from pathlib import Path

import matplotlib.pyplot as plt

from lucas_kanade.tracking_controller import TrackingController
from yolo.helpers.camera import Camera
from yolo.yolo_controller import YoloController

import sys


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen before entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, 'dict'):
        size += get_size(obj.dict, seen)
    elif hasattr(obj, 'iter') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

if __name__ == '__main__':
    yolo = YoloController(cameras=cameras, debug=False)
    lucas_kanade = TrackingController(debug=True)
    i = 0
    while i < 480:
        # print(get_size(cameras))
        # print(get_size(yolo))
        # print(get_size(lucas_kanade))
        i = i + 1
        cars_list = yolo.get_cameras_images()
        lucas_kanade.receiver(cars_list)
        print(i)

    x = list()
    y = list()
    for p in lucas_kanade.car_list[0].positions:
        x.append(p.x)
        y.append(p.y)

    print(len(lucas_kanade.car_list))
    for car in lucas_kanade.car_list:
        print(f'ID - {car.ID}')
    print(len(x))
    plt.plot(x)
    plt.plot(y)
    plt.show()

    # while True:
    #    camera_list.append(yolo.get_cameras_images())
    #    # definir quantidade de frames
    #    time_series = TSConstructor.build_time_series(camera_list)

    # sig = SignalGenerator.generate_random_signal(begin=1, end=10, length=500, plot=True)
    # fft = FastFourierTransformer(signal=sig)
    #
    # fft.fast_transform(plot=True)


