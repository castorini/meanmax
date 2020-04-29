import numpy as np

from .utils import pos_mean_ecdf


def iter_ecdf_distances(ecdf1: np.ndarray,
                        sample1: np.ndarray,
                        ecdf2: np.ndarray,
                        sample2: np.ndarray):
    s1_idx = 0
    s2_idx = 0
    while s1_idx < len(ecdf1) - 1 or s2_idx < len(ecdf2) - 1:
        dist = np.abs(ecdf1[s1_idx] - ecdf2[s2_idx])
        s1 = sample1[s1_idx]
        s2 = sample2[s2_idx]
        yield dist, s1, s2, ecdf1[s1_idx], ecdf2[s2_idx]
        if s1 == s2 and s1_idx != len(ecdf1) - 1 and s2_idx != len(ecdf2) - 1:
            s1_idx += 1
            s2_idx += 1
        elif s1 < s2 and s1_idx != len(ecdf1) - 1:
            s1_idx += 1
        elif s2 < s1 and s2_idx != len(ecdf2) - 1:
            s2_idx += 1
        elif s1_idx == len(ecdf1) - 1:
            s2_idx += 1
        elif s2_idx == len(ecdf2) - 1:
            s1_idx += 1


def compute_ks_distance(ecdf1: np.ndarray,
                        sample1: np.ndarray,
                        ecdf2: np.ndarray,
                        sample2: np.ndarray):
    max_dist = 0
    max_val = 0
    for dist, val, _, _, _ in iter_ecdf_distances(ecdf1, sample1, ecdf2, sample2):
        if dist > max_dist:
            max_dist = dist
            max_val = val
    return max_val, max_dist


def maximum_pointwise_cdf_error(x0, y0):
    if x0 < y0:
        y = x0
        x0 = y0
        y0 = y
    if x0 == 1 and x0 > y0:
        return 1
    k = np.log(np.log(y0) / np.log(x0))
    k = k / np.log(x0 / y0)
    return x0 ** k - y0 ** k


def compute_mme_bias(cdf: np.ndarray,
                     sample: np.ndarray,
                     k: int):
    theta = pos_mean_ecdf(cdf ** k, sample)
    # ethetahat =


def maximum_cdf_error(ecdf1: np.ndarray,
                      sample1: np.ndarray,
                      ecdf2: np.ndarray,
                      sample2: np.ndarray):
    return max(maximum_pointwise_cdf_error(x0, y0) for _, _, _, x0, y0 in iter_ecdf_distances(ecdf1, sample1, ecdf2, sample2))
