""" fft_transformer.py """

import numpy as np
import matplotlib.pyplot as plt
from fft.utils.abstract_vehicle import AbstractVehicle


class FourierController:

    @classmethod
    def build_abstract_vehicle(cls, vehicle_id: str, signal: list) -> AbstractVehicle:
        return AbstractVehicle(vehicle_id, cls._fft(signal))

    @classmethod
    def _fft(cls, signal, **kwargs: bool) -> list:
        """
        This method will perform a Fast Fourier Transform over the instance 'signal' attribute.
        May contain 'plot' keyword argument in case the user wishes to plot the Fourier spectrum.
        :param kwargs: keyword arguments (Optional: ['plot'])
        :return: Fourier spectrum
        """

        points = len(signal)
        fft_calc = np.fft.fft(signal)
        fft_abs = np.abs(fft_calc / points)

        if kwargs.get("plot"):
            plt.plot(fft_abs)
            plt.show()

        return fft_abs
