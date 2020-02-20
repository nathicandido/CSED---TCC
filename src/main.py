""" main.py """

# from fft.fft_transformer import FastFourierTransformer
# from fft.utils.signal_generator import SignalGenerator
from time_series.utils.bounding_box import BoundingBox
from time_series.utils.frame import Frame
from time_series.time_series_constructor import TSConstructor

if __name__ == '__main__':
    frame_list = [
        Frame(index, [BoundingBox(index, index), BoundingBox(index + 1, index + 1)])
        for index in range(10)
    ]

    TSConstructor.build_time_series(frame_list)

    # sig = SignalGenerator.generate_random_signal(begin=1, end=10, length=500, plot=True)
    # fft = FastFourierTransformer(signal=sig)
    #
    # fft.fast_transform(plot=True)
