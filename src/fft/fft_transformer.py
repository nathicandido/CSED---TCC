""" fft_transformer.py """

import numpy as np
import matplotlib.pyplot as plt
from fft.utils.veiculo_abstrato import VeiculoAbstrato


class ControladorFourier:

    @classmethod
    def construir_veiculo_abstrato(cls, id_veiculo, sinal):
        return VeiculoAbstrato(id_veiculo, cls._fft(sinal))

    @classmethod
    def _fft(cls, sinal, **kwargs: bool) -> list:
        pontos = len(sinal)
        fft_calc = np.fft.fft(sinal)
        fft_abs = np.abs(fft_calc / pontos)

        if kwargs.get("plot"):
            plt.plot(fft_abs)
            plt.show()

        return fft_abs
