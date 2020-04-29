from dataclasses import dataclass, field
from typing import Any, Dict

from scipy import stats
import numpy as np

from .estimator import QuantileEstimator
from .utils import compute_pr_x_ge_y
from .tables import MANN_WHITNEY_UP010


@dataclass(frozen=True)
class TwoSampleHypothesisTest(object):
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        raise NotImplementedError


@dataclass(frozen=True)
class StudentsTTest(TwoSampleHypothesisTest):

    @property
    def name(self):
        if not self.options.get('equal_var'):
            return 'Welch\'s t-test'
        else:
            return 't-test'

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        t, p = stats.ttest_ind(sample1, sample2, **self.options)
        return p / 2 < alpha and t < 0, t, p


@dataclass(frozen=True)
class SDBootstrapTest(TwoSampleHypothesisTest):

    @property
    def name(self):
        return 'Stochastic Dominance Bootstrap'

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        iters = self.options.get('iters', 1000)
        gt = compute_pr_x_ge_y(sample1, sample2)
        sample = np.concatenate((sample1, sample2))
        n = len(sample1)
        stats = []
        for _ in range(iters):
            np.random.shuffle(sample)
            sample1 = sample[:n]
            sample2 = sample[n:]
            stats.append(compute_pr_x_ge_y(sample1, sample2))
        p = np.mean(np.array(stats) <= gt)
        return p < alpha, p, p


@dataclass(frozen=True)
class MannWhitneyUTest(TwoSampleHypothesisTest):

    @property
    def name(self):
        return 'Mann-Whitney U test'

    def __post_init__(self):
        if 'alternative' not in self.options:
            self.options['alternative'] = 'less'

    def exact_test(self, s1, s2):
        s1 = [(x, 0) for x in s1]
        s2 = [(x, 1) for x in s2]
        n = len(s1)
        m = len(s2)
        s = sorted(s1 + s2)
        ranksum1 = 0
        ranksum2 = 0
        tmp_ranksum = 0
        n_ranksum = 0
        counts = [0, 0]
        last_x = -1000000
        for rank, (x, l) in enumerate(s):
            if x != last_x and n_ranksum > 0:
                ranksum1 += (tmp_ranksum / n_ranksum) * counts[0]
                ranksum2 += (tmp_ranksum / n_ranksum) * counts[1]
                tmp_ranksum = 0
                n_ranksum = 0
                counts = [0, 0]
            counts[l] += 1
            tmp_ranksum += rank + 1
            n_ranksum += 1
            last_x = x
        if n_ranksum > 0:
            ranksum1 += (tmp_ranksum / n_ranksum) * counts[0]
            ranksum2 += (tmp_ranksum / n_ranksum) * counts[1]
        U1 = (n * m) + (n * (n + 1)) / 2 - ranksum1
        U2 = (n * m) + (m * (m + 1)) / 2 - ranksum2
        U = min(U1, U2)
        return U, 0.05 if U <= MANN_WHITNEY_UP010[n - 1][m - 1] and ranksum1 < ranksum2 else 0.051

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        if len(sample1) <= 20 or len(sample2) <= 20:
            U, p = self.exact_test(sample1, sample2)
        else:
            U, p = stats.mannwhitneyu(sample1, sample2, **self.options)
        return p <= alpha, U, p


@dataclass(frozen=True)
class QuantileTest(TwoSampleHypothesisTest):

    def __post_init__(self):
        if 'quantile' not in self.options:
            self.options['quantile'] = 0.5
        if 'bootstrap_samples' not in self.options:
            self.options['bootstrap_samples'] = 2000
        if 'estimate_method' not in self.options:
            self.options['estimate_method'] = 'harrelldavis'
        if 'alternative' not in self.options:
            self.options['alternative'] = 'less'

    @property
    def name(self):
        if self.options['estimate_method'] == 'harrelldavis':
            return 'Harrell-Davis quantile test'
        if self.options['estimate_method'] == 'direct':
            return 'Direct quantile test'

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        test = QuantileEstimator(dict(estimate_method=self.options['estimate_method'],
                                      quantile=self.options['quantile']))
        dstar_arr = []
        b = self.options['bootstrap_samples']
        for _ in range(b):
            sx = test.estimate_point(np.random.choice(sample1, len(sample1)))
            sy = test.estimate_point(np.random.choice(sample2, len(sample2)))
            dstar_arr.append(sx - sy)
        dstar_arr = np.array(dstar_arr)
        pstar = (sum(dstar_arr < 0) + 0.5 * sum(dstar_arr == 0)) / b
        if self.options['alternative'] == 'less':
            p = 1 - pstar
        elif self.options['alternative'] == 'both':
            p = 2 * min(pstar, 1 - pstar)
        else: # greater
            p = pstar
        return p < alpha, pstar, p


@dataclass(frozen=True)
class ASDTest(TwoSampleHypothesisTest):

    @property
    def name(self):
        return 'Almost Stochastic Dominance test'

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        tmp = sample2
        sample2 = sample1
        sample1 = tmp
        phi = stats.norm.ppf(alpha)
        epsilons = []
        n = len(sample1)
        m = len(sample2)
        c = np.sqrt(n * m / (n + m))
        eps_fn = lambda x, y: 1 - compute_pr_x_ge_y(x, y)
        eps_orig = eps_fn(sample1, sample2)
        for _ in range(1000):
            bs1 = np.random.choice(sample1, n)
            bs2 = np.random.choice(sample2, m)
            epsilons.append(c * (eps_fn(bs1, bs2) - eps_orig))
        min_eps = eps_orig - (1 / c) * np.std(epsilons) * phi
        return min_eps < self.options.get('threshold', 0.5), min_eps, alpha
