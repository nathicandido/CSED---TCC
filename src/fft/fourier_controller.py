""" fourier_controller.py """

import numpy as np
import matplotlib.pyplot as plt
from fft.helpers.abstract_vehicle import AbstractVehicle
from fft.helpers.signal import Signal3D


class FourierController:

    @classmethod
    def build_abstract_vehicle(cls, vehicle_id: str, x_sig: list, y_sig: list) -> AbstractVehicle:
        return AbstractVehicle(vehicle_id, Signal3D(cls._fft(x_sig), cls._fft(y_sig)))

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
