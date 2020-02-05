""" fft_transformer.py """

import numpy as np
import matplotlib.pyplot as plt


class FastFourierTransformer:

    def __init__(self, signal: list = None):
        self.signal = list() if signal is None else signal

    def fast_transform(self, **kwargs: bool) -> list:
        """
        This method will perform a Fast Fourier Transform over the instance 'signal' attribute.
        May contain 'plot' keyword argument in case the user wishes to plot the Fourier spectrum.

        :param kwargs: keyword arguments (Optional: ['plot'])

        :return: Fourier spectrum
        """
        points = len(self.signal)
        fft_calc = np.fft.fft(self.signal)
        fft_abs = 2.0 * np.abs(fft_calc / points)

        if kwargs.get("plot"):
            plt.plot(fft_abs)
            plt.show()

        return fft_abs
