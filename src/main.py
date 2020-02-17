""" main.py """

from src.fft.fft_transformer import FastFourierTransformer
from src.fft.utils.signal_generator import SignalGenerator
from src.time_series.time_series_constructor import TSConstructor
from src.time_series.utils.bounding_box import BoundingBox
from src.time_series.utils.frame import Frame


if __name__ == '__main__':

    bb_list_1 = [
        BoundingBox(1, 2),
        BoundingBox(2, 4),
        BoundingBox(4, 6)
    ]

    bb_list_2 = [
        BoundingBox(4, 5),
        BoundingBox(23, 56),
        BoundingBox(3, 88)
    ]

    bb_list_3 = [
        BoundingBox(12, 56),
        BoundingBox(67, 44),
        BoundingBox(11, 98)
    ]

    frame_list = [Frame(1, bb_list_1), Frame(2, bb_list_2), Frame(3, bb_list_3)]
    TSConstructor.build_time_series(frame_list)

    # sig = SignalGenerator.generate_random_signal(begin=1, end=10, length=500, plot=True)
    # fft = FastFourierTransformer(signal=sig)
    #
    # fft.fast_transform(plot=True)

