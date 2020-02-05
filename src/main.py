""" main.py """

from src.fft.fft_transformer import FastFourierTransformer
from src.fft.utils.signal_generator import SignalGenerator


if __name__ == '__main__':
    sig = SignalGenerator.generate_random_signal(begin=1, end=10, length=500, plot=True)
    fft = FastFourierTransformer(signal=sig)

    print(fft.fast_transform(plot=True))

