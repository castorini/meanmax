from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict

from scipy.stats import beta
from scipy.special import betainc
import numpy as np

from .ci import bootstrap_ci


@dataclass(frozen=True)
class Estimator(object):
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def estimate_point(self, sample: np.ndarray):
        raise NotImplementedError

    def estimate_interval(self, sample: np.ndarray, alpha=0.05):
        return bootstrap_ci(sample,
                            self.estimate_point,
                            alpha=alpha,
                            method='percentile-bootstrap',
                            ci_samples=2000)

    @classmethod
    def gen_class(cls, default_options: Dict[str, Any] = None):
        def new_object(options=None):
            options = dict() if options is None else options.copy()
            options.update(default_options)
            return cls(options)

        if default_options is None:
            return cls
        return new_object


@dataclass(frozen=True)
class JoinedUnbiasedBiasedEstimator(Estimator):

    @property
    def name(self):
        return f'{self.options["estimator1"].name} {self.options["estimator2"].name}'

    def estimate_point(self, sample: np.ndarray):
        var1 = self.options['estimator1_var']


@lru_cache(maxsize=1000)
def beta_cdf_linspace(a, b, n, pow=1):
    inds = np.linspace(0, 1, n + 1) ** pow
    cdfs = beta(a, b).cdf(inds)
    return cdfs[1:] - cdfs[:-1]


@lru_cache(maxsize=1000)
def beta_cdf_shifted(a, b, n, pow=1):
    deltaU = np.array([betainc(a, b, (i / n) ** pow) for i in range(1, n + 1)])
    deltaL = np.array([betainc(a, b, (i / n) ** pow) for i in range(0, n)])
    delta = deltaU - deltaL
    return delta


def harrelldavis_estimate(sample, q, pow=1):
    n = len(sample)
    a = (n + 1) * q
    b = (n + 1) * (1 - q)
    # delta = beta_cdf_linspace(a, b, n, pow=pow)
    sample = np.sort(sample)
    delta = beta_cdf_shifted(a, b, n, pow=pow)
    return np.sum(delta * sample)


@lru_cache(maxsize=1000)
def sorted_cache(a):
    return np.sort(a.x)


@dataclass(frozen=True)
class QuantileEstimator(Estimator):

    def __post_init__(self):
        if 'quantile' not in self.options:
            self.options['quantile'] = 0.5
        if 'ci_method' not in self.options:
            self.options['ci_method'] = 'percentile-bootstrap'
        if 'ci_samples' not in self.options:
            self.options['ci_samples'] = 1000
        if 'estimate_method' not in self.options:
            self.options['estimate_method'] = 'harrelldavis'
        if 'discrete' not in self.options:
            self.options['discrete'] = False

    @property
    def name(self):
        return f'Quantile ({self.options["quantile"]}) {self.options["estimate_method"]} estimator'

    def estimate_point(self, sample: np.ndarray):
        q = self.options['quantile']
        if self.options['estimate_method'] == 'harrelldavis':
            return harrelldavis_estimate(sample, q, discrete=self.options['discrete'])
        elif self.options['estimate_method'] == 'direct':
            return np.quantile(sample, q)

    def estimate_interval(self, sample, alpha=0.05):
        return bootstrap_ci(sample,
                            self.estimate_point,
                            alpha=alpha,
                            method=self.options['ci_method'],
                            ci_samples=self.options['ci_samples'])
