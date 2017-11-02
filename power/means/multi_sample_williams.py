import math
from power import (PowerBase,
                            is_numeric,
                            is_positive,
                            is_boolean)
from power.means import OneSample
from scipy.optimize import brenth
import scipy.stats as stats


class MultiSampleWilliams(PowerBase):
    """ Power and Sample Size calculations for multi-sample Williams Design
    for means.

    The test for equality tests :math:`H_{0}: \\epsilon = 0` versus
    :math:`H_{1}: \\epsilon \\ne 0`.

    The test for superiority tests :math:`H_{0}: \\epsilon > 0` versus
    :math:`H_{1}: \\epsilon \\le 0`.

    The test for equivalence tests :math:`H_{0}: \\epsilon \\ne 0` versus
    :math:`H_{1}: \\epsilon = 0`.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        mu: :math:`\\mu` is a list of means to be tested.
        stdev: :math:`\\sigma` is the standard deviation of the sample.
        known_stdev: A boolean indicator if the standard deviation.
        hypothesis: One of 'equality', 'superiority', or 'equvalence'.  Tests
            of 'non-inferiority' are the same as tests of 'superiority' with
            respect to the power calculation so choose 'superiority' for both.
        margin: This is the superiority or equivalence margin.
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).

    Note:
        Currently, this function returns the best power and the best alpha when
        power or alpha are unknown.  I don't think this is the best approach,
        so it may change.  I may change it to return a power calculation for
        each contrast.
    """

    def __init__(self, n=None, mu=None, stdev=None, hypothesis=None,
                 margin=None, alpha=None, beta=None, power=None,
                 known_stdev=None):
        if n is not None:
            if isinstance(n, int):
                self.n = n
            else:
                raise ValueError('`n` should be of type Int.')
        else:
            self.n = None

        for value in mu:
            is_numeric(value, 'value')
        mu = [float(value) for value in mu]
        self.mu = mu
        self.n_groups = len(self.mu)
        if self.n_groups % 2 == 0:
            # If the number of groups is even, the Williams design is
            # equivalent to a n_groups by n_groups crossover design
            self.k = self.n_groups
        else:
            # Otherwise, it is equal to a 2*n_groups by n_groups crossover
            # design
            self.k = self.n_groups * 2

        is_positive(stdev, 'stdev')
        self.stdev = stdev

        if known_stdev is not None:
            is_boolean(known_stdev, 'known_stdev')
        else:
            known_stdev = True
        self.known_stdev = known_stdev

        if margin is not None:
            is_numeric(margin, 'margin')
        else:
            margin = 0
        self.margin = float(margin)

        # Initialize the remaining arguments (alpha, beta, power)
        # through the parent.
        super(MultiSampleWilliams, self).__init__(alpha=alpha,
                                                  power=power,
                                                  beta=beta,
                                                  hypothesis=hypothesis)

    @staticmethod
    def _calculate_power_unknown(n, alpha, theta, n_groups, _alpha_adjustment):
        """ Calculate power in the case that stdev is unknown.

        This is an internal static method only.  This overrides the
        superclasses method.
        """
        nu = n_groups * (n - 1)
        ncp = math.sqrt(n) * abs(theta)
        quantile = stats.t.ppf(1 - alpha / _alpha_adjustment, nu)
        power = (1 - stats.nct.cdf(quantile, nu, ncp) +
                 stats.nct.cdf(-1 * quantile, nu, ncp))
        return power

    def calculate(self):
        """ Performs the power calculation """
        if self.known_stdev:
            # When stdev is known, this collapses to a one sample test with
            # stdev = stdev / sqrt(k)
            if self.n is None:
                n = 0
                power = None
                for i in range(0, self.n_groups - 1):
                    for j in range(i + 1, self.n_groups):
                        epsilon = self.mu[i] - self.mu[j]
                        one_sample = OneSample(mu=self.mu[i],
                                               mu_0=self.mu[j],
                                               margin=self.margin,
                                               hypothesis=self.hypothesis,
                                               stdev=(self.stdev /
                                                      math.sqrt(self.k)),
                                               known_stdev=self.known_stdev,
                                               alpha=self.alpha,
                                               power=self.power)
                        one_sample.calculate()
                        if one_sample.n > n:
                            n = one_sample.n
                            power = one_sample.power
                self.n = n
                self.power = power
                self.beta = 1 - self.power
            elif self.power is None:
                power = 0
                for i in range(0, self.n_groups - 1):
                    for j in range(i + 1, self.n_groups):
                        epsilon = self.mu[i] - self.mu[j]
                        one_sample = OneSample(n=self.n,
                                               mu=self.mu[i],
                                               mu_0=self.mu[j],
                                               margin=self.margin,
                                               hypothesis=self.hypothesis,
                                               stdev=(self.stdev /
                                                      math.sqrt(self.k)),
                                               known_stdev=self.known_stdev,
                                               alpha=self.alpha)
                        one_sample.calculate()
                        if one_sample.power > power:
                            power = one_sample.power
                self.power = power
                self.beta = 1 - self.power
            elif self.alpha is None:
                alpha = 1
                for i in range(0, self.n_groups - 1):
                    for j in range(i + 1, self.n_groups):
                        epsilon = self.mu[i] - self.mu[j]
                        one_sample = OneSample(n=self.n,
                                               mu=self.mu[i],
                                               mu_0=self.mu[j],
                                               margin=self.margin,
                                               hypothesis=self.hypothesis,
                                               stdev=(self.stdev /
                                                      math.sqrt(self.k)),
                                               known_stdev=self.known_stdev,
                                               power=self.power)
                        one_sample.calculate()
                        if one_sample.alpha < alpha:
                            alpha = one_sample.alpha
                self.alpha = alpha
        else:
            # When stdev is unknown, this should collapse to a one sample test
            # but the degrees of freedom are different.
            if self.n is None:
                n = 0
                power = None
                for i in range(0, self.n_groups - 1):
                    for j in range(i + 1, self.n_groups):
                        epsilon = abs(self.mu[i] - self.mu[j])
                        if self.hypothesis is 'superiority':
                            epsilon = epsilon - self.margin
                        elif self.hypothesis is 'equivalence':
                            epsilon = self.margin - abs(epsilon)
                        theta = epsilon * math.sqrt(self.k) / self.stdev
                        res = brenth(lambda n:
                                     self._calculate_power_unknown(n,
                                                                   self.alpha,
                                                                   theta,
                                                                   self._alpha_adjustment,
                                                                   self._beta_adjustment) -
                                     self.power, a=2, b=1e7)
                        test_n = math.ceil(res)
                        test_power = self._calculate_power_unknown(test_n,
                                                                   self.alpha,
                                                                   theta,
                                                                   self._alpha_adjustment,
                                                                   self._beta_adjustment)
                        if test_n > n:
                            n = test_n
                            power = test_power
                self.n = n
                self.power = power
            elif self.power is None:
                power = 0
                for i in range(0, self.n_groups - 1):
                    for j in range(i + 1, self.n_groups):
                        epsilon = abs(self.mu[i] - self.mu[j])
                        if self.hypothesis is 'superiority':
                            epsilon = epsilon - self.margin
                        elif self.hypothesis is 'equivalence':
                            epsilon = self.margin - abs(epsilon)
                        theta = epsilon * math.sqrt(self.k) / self.stdev
                        test_power = self._calculate_power_unknown(self.n,
                                                                   self.alpha,
                                                                   theta,
                                                                   self._alpha_adjustment,
                                                                   self._beta_adjustment)
                        if test_power > power:
                            power = test_power
                self.power = power
                self.beta = 1 - self.power
            elif self.alpha is None:
                alpha = 1
                for i in range(0, self.n_groups - 1):
                    for j in range(i + 1, self.n_groups):
                        epsilon = abs(self.mu[i] - self.mu[j])
                        if self.hypothesis is 'superiority':
                            epsilon = epsilon - self.margin
                        elif self.hypothesis is 'equivalence':
                            epsilon = self.margin - abs(epsilon)
                        theta = epsilon * math.sqrt(self.k) / self.stdev
                        res = brenth(lambda alpha:
                                     self._calculate_power_unknown(self.n,
                                                                   alpha,
                                                                   theta,
                                                                   self._alpha_adjustment,
                                                                   self._beta_adjustment) -
                                     self.power,
                                     a=0, b=1)
                        if res < alpha:
                            alpha = res
                self.alpha = self.tau * one_sample.alpha
