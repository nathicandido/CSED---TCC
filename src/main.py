""" main.py """

from matplotlib import pyplot as plt

from time_series.time_series_constructor import TSConstructor
# from fft.fft_transformer import FastFourierTransformer
# from fft.utils.signal_generator import SignalGenerator
from time_series.utils.bounding_box import BoundingBox
from time_series.utils.frame import Frame

if __name__ == '__main__':
    frame_list = [
        Frame(index, [BoundingBox(index + 1, index), BoundingBox(index, index + 1)])
        for index in range(50)
    ]

    time_series = TSConstructor.build_time_series(frame_list)

    plt.plot(time_series['0']['pos_x'])
    plt.plot(time_series['0']['pos_y'])

    plt.show()

    plt.plot(time_series['1']['pos_x'])
    plt.plot(time_series['1']['pos_y'])

    plt.show()

    # sig = SignalGenerator.generate_random_signal(begin=1, end=10, length=500, plot=True)
    # fft = FastFourierTransformer(signal=sig)
    #
    # fft.fast_transform(plot=True)
