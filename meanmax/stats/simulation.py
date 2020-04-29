from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Sequence, Callable, Tuple

from tqdm import trange
import numpy as np
import statsmodels.api as sm

from .test import TwoSampleHypothesisTest
from .utils import compute_pr_x_ge_y


BinaryOrderingFn = Callable[[np.ndarray, np.ndarray], float]


def dfromc_rvs(bins, cdf, **cdf_kwargs):
    xs = np.linspace(0, 1, bins + 1)
    cdf_gen = partial(cdf, **cdf_kwargs)
    probs = cdf_gen(xs[1:]) - cdf_gen(xs[:-1])
    probs = probs / probs.sum()
    return partial(np.random.choice, np.linspace(0, 1, bins), p=probs)


def dkde_from_sample_rvs(sample: np.ndarray):
    kde = sm.nonparametric.KDEUnivariate(sample)
    kde.fit()
    pmf = kde.cdf[1:] - kde.cdf[:-1]
    pmf = pmf / pmf.sum()
    return partial(np.random.choice, kde.support[:-1], p=pmf)


def order_stochastic(d1, d2):
    return compute_pr_x_ge_y(d1, d2) - 0.5


def order_quantile(d1, d2, quantile=0.5):
    return np.quantile(d1, quantile) - np.quantile(d2, quantile)


@dataclass
class ResultPopulationPair(object):
    pop_x: np.ndarray
    pop_y: np.ndarray

    def compute_pr_x_ge_y(self):
        return compute_pr_x_ge_y(self.pop_x, self.pop_y)

    def simulate_statistical_power(self,
                                   n1,
                                   tests: Sequence[TwoSampleHypothesisTest],
                                   use_tqdm=False,
                                   order_fn: BinaryOrderingFn = order_stochastic,
                                   **kwargs):
        pop_small, pop_big = (self.pop_x, self.pop_y) if order_fn(self.pop_x, self.pop_y) <= 0 else (self.pop_y, self.pop_x)
        iters = kwargs.get('iters', 500)
        alpha = kwargs.get('alpha', 0.05)
        n2 = kwargs.get('n2', n1)

        counter = Counter()
        for _ in trange(iters, disable=not use_tqdm):
            sx = np.random.choice(pop_small, n1, replace=False)
            sy = np.random.choice(pop_big, n2, replace=False)

            for test in tests:
                reject, stat, p = test.test(sx, sy, alpha=alpha)
                counter[test.name] += int(reject)
        return {k: v / iters for k, v in counter.items()}


@dataclass
class ResultPopulationSingle(object):
    pop: np.ndarray

    def simulate_type1_error(self, n, tests: Sequence[TwoSampleHypothesisTest], use_tqdm=False, **kwargs):
        iters = kwargs.get('iters', 500)
        alpha = kwargs.get('alpha', 0.05)

        counter = Counter()
        for _ in trange(iters, disable=not use_tqdm):
            for test in tests:
                sx1 = np.random.choice(self.pop, n, replace=False)
                sx2 = np.random.choice(self.pop, n, replace=False)
                reject, stat, p = test.test(sx1, sx2, alpha=alpha)
                counter[test.name] += int(reject)
        return {k: v / iters for k, v in counter.items()}
