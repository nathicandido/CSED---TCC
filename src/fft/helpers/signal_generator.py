""" signal_generator.py """

from random import uniform
from typing import Union
import matplotlib.pyplot as plt

from utils.exceptions.exceptions import MissingArgumentsError


class SignalGenerator:

    @classmethod
    def generate_random_signal(cls, **kwargs: Union[int, bool]) -> list:
        """
        This method will generate a random signal. It must receive the keyword arguments 'begin' and 'end', which
        will be responsible for building a range of numbers which could be generated. Also,
        it is necessary to specify the 'length' of the signal. If any of these arguments are nowhere
        to be found, a error will be raised

        :param kwargs: keyword arguments (Obligatory: ['begin', 'end', 'length']; Optional: ['plot'])
        :return: randomly generated signal
        """

        begin = kwargs.get("begin")
        end = kwargs.get("end")
        length = kwargs.get("length")
        plot = kwargs.get("plot")

        if not all([begin, end, length]):
            raise MissingArgumentsError("Arguments are missing")

        signal = [uniform(begin, end) for _ in range(length)]

        if plot:
            plt.plot(signal)
            plt.show()

        return signal
