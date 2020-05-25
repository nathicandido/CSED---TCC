""" fourier_controller.py """

import numpy as np
import matplotlib.pyplot as plt
from fft.helpers.abstract_vehicle import AbstractVehicle
from fft.helpers.signal import Signal3D
from fft.helpers.interpolator import Interpolator


class FourierController:

    @classmethod
    def build_abstract_vehicle(cls, vehicle_id: str, x_sig: list, y_sig: list) -> AbstractVehicle:
        return AbstractVehicle(
            vehicle_id,
            Signal3D(
                cls.smoothen_and_interpolate(x_sig),
                cls.smoothen_and_interpolate(y_sig)
            )
        )

    @classmethod
    def smoothen_and_interpolate(cls, signal, **kwargs: bool) -> np.ndarray:
        """
        This method will perform a Fast Fourier Transform over the instance 'signal' attribute.
        May contain 'plot' keyword argument in case the user wishes to plot the Fourier spectrum.
        :param signal:
        :param kwargs: keyword arguments (Optional: ['plot'])
        :return: Fourier spectrum
        """

        rfft_calc = np.fft.rfft(signal)
        rfft_calc[7:] = 0
        smoothened_ts = np.fft.irfft(rfft_calc, len(signal))
        interpolated_ts = Interpolator.interpolate_time_series(smoothened_ts)

        if kwargs.get("plot"):
            plt.plot(interpolated_ts)
            plt.show()

        return interpolated_ts
