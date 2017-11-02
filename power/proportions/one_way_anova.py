from power import (PowerBase,
                            is_in_0_1,
                            is_integer)
from power.means import OneSample
import statistics
import math


class OneWayAnova(PowerBase):
    """ Power and Sample Size calculations for one way ANOVA.

    One Way ANOVA uses the following model:

    :math:`x_{ij} = p_i + \\epsilon_{ij}`

    where :math:`p_i` is the fixed effect of the :math:`i^{\\textbf{th}}`
    treatment and :math:`\\epsilon_{ij}` is the random error from observing
    :math:`x_{ij}`.

    The hypothesis of interest may either be a pairwise comparison, where the
    hypotheses are:

    :math:`H_{0}: p_i > p_j`

    versus

    :math:`H_{1}: p_i \\ne p_j`

    for some pair :math:`(i, j)`, or a simultaneous comparison, where the
    hypotheses are:

    :math:`H_{0}: p_1 = p_2 = \\dots = p_k`

    versus

    :math:`H_{1}: p_i \\ne p_j`

    for some :math:`1 \\le i < j \\le k`.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        p: A list of proportions of the form
            [:math:`p_1`, `p_2`, ..., `p_k`]
        p_0: A null proportion to test against.  If provided, the test is done
            between each value in p.  Otherwise it is done pairwise within p.
        sigma: :math:`\\sigma` is estimated the standard deviation of the
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
    def __init__(self, n=None, p=None, p_0=None, alpha=None,
                 beta=None, power=None):
        if n is not None:
            if isinstance(n, int):
                self.n = n
            else:
                raise ValueError('`n` should be of type Int.')
        else:
            self.n = None

        for value in p:
            is_in_0_1(value, 'Each value in `p` should be in [0, 1].')

        if p_0 is not None:
            is_in_0_1(value, '`p_0` should be in [0, 1].')

        self.p = p
        self.p_0 = p_0
        self.mean_mu = statistics.mean(self.p)
        self.n_groups = len(self.p)

        # The number of comparisons of interest for pairwise comparisons
        self.tau = self.n_groups * (self.n_groups - 1) / 2.0

        # Initialize the remaining arguments (alpha, beta, power)
        # through the parent.
        super(OneWayAnova, self).__init__(alpha=alpha, power=power,
                                          beta=beta, hypothesis='equality')

    def calculate(self):
        """ Perfrom the power calculation """
        # Adjust alpha for multiple comparisons
        alpha = self.alpha / (2 * self.tau)
        if self.n is None:
            n = 0
            power = None
            if self.p_0 is None:
                for i in range(0, self.n_groups - 1):
                    for j in range(i + 1, self.n_groups):
                        stdev = math.sqrt(self.p[i] * (1 - self.p[i]) +
                                          self.p[j] * (1 - self.p[j]))
                        one_sample = OneSample(mu=self.p[i],
                                               mu_0=self.p[j],
                                               stdev=stdev,
                                               known_stdev=True,
                                               alpha=alpha,
                                               power=self.power)
                        one_sample.calculate()
                        if one_sample.n > n:
                            n = one_sample.n
                            power = one_sample.power
            else:
                for i in range(0, self.n_groups):
                    stdev = math.sqrt(self.p[i] * (1 - self.p[i]) +
                                      self.p_0 * (1 - self.p_0))
                    one_sample = OneSample(mu=self.p[i],
                                           mu_0=self.p_0,
                                           stdev=stdev,
                                           known_stdev=True,
                                           alpha=alpha,
                                           power=self.power)
                    one_sample.calculate()
                    if one_sample.n > n:
                        n = one_sample.n
                        power = one_sample.power
            self.n = n
            self.power = power
            self.beta = 1 - self.power
