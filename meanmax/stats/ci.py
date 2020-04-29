import numpy as np

from .utils import ecdf


def bootstrap_ci(sample: np.ndarray,
                 estimate_fn,
                 alpha=0.05,
                 method='percentile-bootstrap',
                 ci_samples=2000):
    est = estimate_fn(sample)
    bs_estimates = [estimate_fn(np.random.choice(sample, len(sample))) for _ in range(ci_samples)]
    if method == 'percentile-bootstrap':
        qa1, qa2 = np.quantile(bs_estimates, (alpha / 2, 1 - alpha / 2))
    elif method == 'reverse-bootstrap':
        qa1, qa2 = np.quantile(bs_estimates, (alpha / 2, 1 - alpha / 2))
        tmp = qa1
        qa1 = 2 * est - qa2
        qa2 = 2 * est - tmp
    return est, (qa1, qa2)


def compute_ecdf_ci_bands(sample: np.ndarray, alpha, k=1, **kwargs):
    n = len(sample)
    eps = np.sqrt(np.log(2 / alpha) / (2 * n)) ** k
    cdf, sample = ecdf(sample, **kwargs)
    cdf = cdf ** k
    return (np.clip(cdf - eps, 0, 1), np.clip(cdf + eps, 0, 1)), sample
