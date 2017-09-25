import numpy as np


def dft_factor(k, n, n_tot, sampling_rate):
    return np.exp(2j * np.pi * k * n / n_tot)


def dft_matrix(n_tot, sampling_rate):
    ret_mat = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for k in range(0, n_tot):
        for n in range(0, n_tot):
            ret_mat[k, n] = dft_factor(k, n, n_tot, sampling_rate)
    return ret_mat


def idft_matrix(n_tot, sampling_rate):
    ret_mat = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for k in range(0, n_tot):
        for n in range(0, n_tot):
            ret_mat[k, n] = dft_factor(k, n, n_tot, sampling_rate).conjugate()
    return ret_mat


# gives angular frequency -> 2 pi nu = omega
def dft_frequencies(n_tot, sampling_rate):
    return np.linspace(0, 2 * np.pi * (n_tot - 1) * sampling_rate / n_tot, n_tot, dtype=np.float64)


def idft_times(n_tot, sampling_rate):
    return np.linspace(0, (n_tot - 1) / sampling_rate, n_tot, dtype=np.float64)


def dft(f_array, sampling_rate):
    return np.array(
        [dft_frequencies(len(f_array), sampling_rate),
         np.dot(dft_matrix(len(f_array), sampling_rate), f_array) / np.sqrt(len(f_array))])


def idft(f_array, sampling_rate):
    return np.array([idft_times(len(f_array), sampling_rate),
                     idft_matrix(len(f_array), sampling_rate).dot(f_array) / np.sqrt(len(f_array))])
