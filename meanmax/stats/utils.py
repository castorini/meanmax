import numpy as np


def compute_pr_x_ge_y(x, y):
    x = [(x_, 0) for x_ in x]
    y = [(y_, 1) for y_ in y]
    z = sorted(y + x, key=lambda e: e[0])
    counter = 0
    results = []
    for idx, (elem, lab) in enumerate(z):
        counter += lab
        results.append(counter * (1 - lab))
    return np.sum(results) / (len(x) * len(y))


def pos_mean_ecdf(cdf, sample):
    cdf1 = 1 - cdf
    min_sample = sample[0]
    a = min_sample
    delta = sample[1:] - sample[:-1]
    return a + np.sum(cdf1[:-1] * delta)


def ecdf(sample: np.ndarray,
         equality=True,
         sorted=True,
         start_point=False,
         return_map=False):
    cdf = []
    if start_point: cdf.append(0)
    if not sorted:
        sample = np.sort(sample)
    n = len(sample)
    next_sample = np.concatenate((sample[1:], [None]))
    last_prob = 0
    for idx, (s, next_s) in enumerate(zip(sample, next_sample)):
        if s != next_s:
            prob = (idx + 1) / n if equality else last_prob
            last_prob = (idx + 1) / n
            cdf.append(prob)
    if equality:
        cdf.append(1)
    sample = np.unique(sample)
    cdf = np.unique(cdf)
    if return_map:
        return {s: c for s, c in zip(sample, cdf)}
    return cdf, sample


def compute_minimum_sample_power(max_p, alpha=0.95):
    return np.log(1 - alpha) / np.log(1 - max_p)


if __name__ == '__main__':
    S = np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4])
    n = len(S)
    cdf, _ = ecdf(S, equality=True)
    cdf_less, S = ecdf(S, equality=False)
    print(cdf, cdf_less)
    print('=' * 50)
    print(np.sum(cdf - cdf_less))
    print(np.sum((cdf ** n - cdf_less ** n) * S))