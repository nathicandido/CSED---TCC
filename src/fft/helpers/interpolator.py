from scipy.interpolate import interp1d
from numpy import linspace


class Interpolator:

    @classmethod
    def interpolate_time_series(cls, ts):
        interp_func = interp1d(linspace(0, len(ts), len(ts)), ts, kind='cubic')
        xnew = linspace(0, len(ts), 100)
        return interp_func(xnew)