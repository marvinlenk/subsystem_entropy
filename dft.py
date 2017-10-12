import numpy as np


def dft_factor(k, n, n_tot):
    return np.exp(2j * np.pi * k * n / n_tot)


def dft_matrix(n_tot):
    ret_mat = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for k in range(0, n_tot):
        for n in range(0, n_tot):
            ret_mat[k, n] = dft_factor(k, n, n_tot)
    return ret_mat


def idft_matrix(n_tot):
    ret_mat = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for k in range(0, n_tot):
        for n in range(0, n_tot):
            ret_mat[k, n] = dft_factor(k, n, n_tot).conjugate()
    return ret_mat


# gives angular frequency -> 2 pi nu = omega
def dft_frequencies(n_tot, sampling_rate):
    tmp = np.linspace(0, np.pi * (n_tot - 1) * sampling_rate / n_tot, int(n_tot / 2) + 1, dtype=np.float64)
    tmp2 = np.linspace(-np.pi * (n_tot - 1) * sampling_rate / n_tot, 0, int(n_tot / 2) + 1, dtype=np.float64)
    return np.append(tmp, tmp2[:-1])


def idft_times(n_tot, sampling_rate):
    return np.linspace(0, (n_tot - 1) / sampling_rate, n_tot, dtype=np.float64)


def dft(f_array, sampling_rate):
    return np.column_stack((dft_frequencies(len(f_array), sampling_rate),
                            np.dot(dft_matrix(len(f_array)), f_array) / np.sqrt(len(f_array))))


def idft(f_array, sampling_rate):
    return np.column_stack((idft_times(len(f_array), sampling_rate),
                            idft_matrix(len(f_array)).dot(f_array) / np.sqrt(len(f_array))))

def rearrange(f_array):
    return np.concatenate((f_array[int(len(f_array)/2) + 1:], f_array[:int(len(f_array)/2) + 1]))
