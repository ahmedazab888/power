import statistics
import math
from power import (PowerBase,
                            is_numeric,
                            is_positive,
                            is_boolean)
from power.means import OneSample
from scipy.optimize import brenth
import scipy.stats as stats


class OneWayAnova(PowerBase):
    """ Power and Sample Size calculations for one way ANOVA.

    One Way ANOVA uses the following model:

    :math:`x_{ij} = \\mu_i + \\epsilon_{ij}`

    where :math:`\\mu_i` is the fixed effect of the :math:`i^{\\textbf{th}}`
    treatment and :math:`\\epsilon_{ij}` is the random error from observing
    :math:`x_{ij}`.

    The hypothesis of interest may either be a pairwise comparison, where the
    hypotheses are:

    :math:`H_{0}: \\mu_i > \\mu_j`

    versus

    :math:`H_{1}: \\mu_i \\ne \\mu_j`

    for some pair :math:`(i, j)`, or a simultaneous comparison, where the
    hypotheses are:

    :math:`H_{0}: \\mu_1 = \\mu_2 = \\dots = \\mu_k`

    versus

    :math:`H_{1}: \\mu_i \\ne \\mu_j`

    for some :math:`1 \\le i < j \\le k`.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        mu: A list of means of the form
            [:math:`\\mu_1`, `\\mu_2`, ..., `\\mu_k`]
        stdev: :math:`\\sigma` is estimated the standard deviation of the
            sample.
        known_stdev: A boolean indicator if the standard deviation is known.
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        comparison: A string from ['simultaneous', 'pairwise'] indicating if
            the simultaneous or pairwise hypothesis is of interest.  The
            default is pairwise.
    """
    def __init__(self, n=None, mu=None, stdev=None, comparison=None,
                 known_stdev=None, alpha=None, beta=None, power=None):
        if n is not None:
            if isinstance(n, int):
                self.n = n
            else:
                raise ValueError('`n` should be of type Int.')
        else:
            self.n = None

        for value in mu:
            is_numeric(value, 'value')
        self.mu = mu
        self.mean_mu = statistics.mean(self.mu)
        self.n_groups = len(self.mu)

        # The number of comparisons of interest for pairwise comparisons
        self.tau = self.n_groups * (self.n_groups - 1) / 2.0

        is_positive(stdev, 'stdev')
        self.stdev = stdev

        if comparison in ['simultaneous', 'pairwise']:
            self.comparison = comparison
        elif comparison is None:
            self.comparison = 'pairwise'
        else:
            raise ValueError(("`comparison` should be in ['simultaneous', "
                              "'pairwise']"))

        if known_stdev is not None:
            is_boolean(known_stdev, 'known_stdev')
        else:
            known_stdev = True
        self.known_stdev = known_stdev

        # Initialize the remaining arguments (alpha, beta, power)
        # through the parent.
        super(OneWayAnova, self).__init__(alpha=alpha, power=power,
                                          beta=beta, hypothesis='equality')

    def calculate_delta(self):
        """ Calculate the mean squared deviation for the values in mu """
        summand = 0
        for value in self.mu:
            summand += (value - self.mean_mu)**2
        delta = summand / self.stdev**2
        return delta

    @staticmethod
    def _calculate_simultaneous_power(ncp, alpha, n_groups):
        """ Calculate power in the case that simultaneous comparisons are used

        This is an internal static method only.
        """
        nu = n_groups - 1
        quantile = stats.chi2.ppf(1 - alpha, nu)
        power = (1 - stats.ncx2.cdf(quantile, nu, ncp))
        return power

    def calculate(self):
        """ Performs the power calculation """
        if self.comparison == 'pairwise':
            self.calculate_pairwise()
        else:
            self.calculate_simultaneous()

    def calculate_pairwise(self):
        """ Performs the power calculation for pairwise comparisions """
        # Adjust alpha for multiple comparisons
        alpha = self.alpha / (self.tau * 2)
        if self.n is None:
            n = 0
            power = None
            for i in range(0, self.n_groups - 1):
                for j in range(i + 1, self.n_groups):
                    one_sample = OneSample(mu=self.mu[i],
                                           mu_0=self.mu[j],
                                           stdev=self.stdev,
                                           known_stdev=self.known_stdev,
                                           alpha=alpha,
                                           power=self.power)
                    one_sample.calculate()
                    if one_sample.n > n:
                        n = one_sample.n
                        power = one_sample.power
            self.n = n
            self.power = power
            self.beta = 1 - self.power
        elif self.power is None:
            one_sample = OneSample(n=self.n,
                                   mu=self.mu[i],
                                   mu_0=self.mu[j],
                                   stdev=self.stdev,
                                   known_stdev=self.known_stdev,
                                   alpha=alpha)
            one_sample.calculate()
            self.power = one_sample.power
            self.beta = 1 - self.power
        elif self.alpha is None:
            one_sample = OneSample(n=self.n,
                                   mu=self.mu[i],
                                   mu_0=self.mu[j],
                                   stdev=self.stdev,
                                   known_stdev=self.known_stdev,
                                   power=self.power)
            one_sample.calculate()
            self.alpha = (2 * self.tau) * one_sample.alpha

    def calculate_simultaneous(self):
        """ Performs the power calculation for simultaneous comparisions """
        delta = self.calculate_delta()
        if self.n is None:
            res = brenth(lambda ncp:
                         self._calculate_simultaneous_power(ncp,
                                                            self.alpha,
                                                            self.n_groups) - self.power,
                         a=2, b=1e7)
            self.n = math.ceil(res / delta)
            ncp = delta * self.n
            self.power = self._calculate_simultaneous_power(ncp,
                                                            self.alpha,
                                                            self.n_groups)
            self.beta = 1 - self.power
        elif self.power is None:
            ncp = delta * self.n
            self.power = self._calculate_simultaneous_power(ncp,
                                                            self.alpha,
                                                            self.n_groups)
            self.beta = 1 - self.power
        elif self.alpha is None:
            ncp = delta * self.n
            res = brenth(lambda alpha:
                         self._calculate_simultaneous_power(ncp,
                                                            alpha,
                                                            self.n_groups) - self.power,
                         a=0, b=1)
            self.alpha = res
