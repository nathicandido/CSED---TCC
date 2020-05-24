import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def get_mean_array(cons):
    mean_arr_x = list()
    mean_arr_y = list()

    for c in cons:
        mean_arr_x.append(sum(c[0]) / len(c[0]))
        mean_arr_y.append(sum(c[1]) / len(c[1]))

    return mean_arr_x, mean_arr_y

def plot_mean_x_with_gaussian_noise(mean_arr_x):
    for a in arrays:
        plt.plot(a['x_sig'])

    gaussian_noise_x = [[x + n for x, n in zip(mean_arr_x, np.random.normal(0, 1, 100))] for _ in range(100)]

    for g in gaussian_noise_x:
        plt.plot(smooth_and_interp(g, freq_cut=7))


def plot_mean_y_with_gaussian_noise(mean_arr_y):
    for arr in arrays:
        plt.plot(arr['y_sig'], '--')

    gaussian_noise_y = [[y + n for y, n in zip(mean_arr_y, np.random.normal(0, 2, 100))] for _ in range(100)]

    for g in gaussian_noise_y:
        plt.plot(smooth_and_interp(g, freq_cut=7))


def generate_gaussian_noise(n, scale, signal_):
    gaussian_noise = [[p + n for p, n in zip(signal_, np.random.normal(0, scale, 100))] for _ in range(n)]
    return gaussian_noise

def interpolate_time_series(ts):
    interp_func = interp1d(np.linspace(0, len(ts), len(ts)), ts, kind='cubic')
    xnew = np.linspace(0, len(ts), 100)
    return interp_func(xnew)

def smooth_and_interp(signal_, freq_cut):
    rfft_calc = np.fft.rfft(signal_)
    rfft_calc[freq_cut:] = 0
    smoothened_ts = np.fft.irfft(rfft_calc, len(signal_))
    interpolated_ts = interpolate_time_series(smoothened_ts)

    return interpolated_ts


def absolute_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            yield os.path.abspath(os.path.join(dirpath, file))

if __name__ == '__main__':


    arrays = list()

    for index, f in enumerate(absolute_file_paths('C:/Users/Tiagoo/PycharmProjects/CSED---TCC/dataset/parsed')):
        arrays.append(np.load(f))
        print(f'{index} - {f}')

    x_sig_group = map(lambda a: a['x_sig'], arrays)
    y_sig_group = map(lambda a: a['y_sig'], arrays)

    x_pos_group = [pos for pos in zip(*x_sig_group)]
    y_pos_group = [pos for pos in zip(*y_sig_group)]

    consolidate = [(x, y) for x, y in zip(x_pos_group, y_pos_group)]

    mean_x, mean_y = get_mean_array(consolidate)

    gaussian_x = [generate_gaussian_noise(100, .5, x) for x in x_sig_group]
    gaussian_y = [generate_gaussian_noise(100, 1, y) for y in y_sig_group]

    sm_gaussian_x = [[smooth_and_interp(g, freq_cut=7) for g in matrix] for matrix in gaussian_x]
    sm_gaussian_y = [[smooth_and_interp(g, freq_cut=7) for g in matrix] for matrix in gaussian_y]

    for index, (x_matrix, y_matrix) in enumerate(zip(sm_gaussian_x, sm_gaussian_y)):
        for x_signal, y_signal in zip(x_matrix, y_matrix):
            plt.plot(x_signal)
            plt.plot(y_signal)

        plt.title(index)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

    gaussian_y = [generate_gaussian_noise(100, 15, y) for y in y_sig_group]

    plot_mean_x_with_gaussian_noise(mean_x)
    plot_mean_y_with_gaussian_noise(mean_y)

    plt.plot(mean_x, 'o', label='Mean X')
    plt.plot(mean_y, 'o', label='Mean Y')
    plt.legend(loc='best')
    plt.show()

