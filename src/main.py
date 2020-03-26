""" main.py """

from matplotlib import pyplot as plt

from time_series.time_series_constructor import TSConstructor
# from fft.fft_transformer import FastFourierTransformer
# from fft.utils.signal_generator import SignalGenerator
from time_series.utils.bounding_box import BoundingBox
from time_series.utils.frame import Frame
from yolo.utils.camera import Camera
from yolo.yolo import Yolo

if __name__ == '__main__':
    cameras = [Camera(0)]
    yolo = Yolo(cameras)

    camera_list = list()

    while True:
        camera_list.append(yolo.get_cameras_images())
        # definir quantidade de frames
        time_series = TSConstructor.build_time_series(camera_list)



    # sig = SignalGenerator.generate_random_signal(begin=1, end=10, length=500, plot=True)
    # fft = FastFourierTransformer(signal=sig)
    #
    # fft.fast_transform(plot=True)
