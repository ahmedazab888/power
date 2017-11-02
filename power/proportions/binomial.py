import math
from power import (PowerBase,
                            is_in_0_1,
                            is_integer)
import scipy.stats as stats
import numpy.random as random


class Binomial(PowerBase):
    """ Hypotheses for a Binomial test of proportions.

    Hypothesis:
        The test for equality and the test for superiority can be unified under
        the hypothesis:

        :math:`H_{0}: p = p_0 + \\delta`
        :math:`H_{1}: p = p_1`

        where :math:`p_0` is the null value, :math:`p_1` is the estimated value
        of the probability and :math:`\\delta` is the margin.

        Note that when :math:`\\delta = 0`, the hypothesis is an equality test.
        :math:`\\delta > 0 (< 0)` corresponds to a superiority
        (non-inferiority) test.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        p_0: The null value for the probability
        p_1: The estimated value for the probability
        margin: The margin used in superiority hypotheses
    """

    # Parameters controling the simulation of power
    _N_SIMS = 1000
    _SEED = 701321

    # Parameters controling the search grid for the calculation of sample size.
    _minN = 2
    _maxN = 100

    def __init__(self, n=None, alpha=None, beta=None, power=None,
                 p=None, p_0=None, margin=None):

        is_in_0_1(p, 'p')
        self.p = p

        is_in_0_1(p_0, 'p_0')
        self.p_0 = p_0

        if margin is not None:
            is_in_0_1(margin + p_0, 'margin + p_0')
            self.margin = margin
            self.p_0 += margin
        else:
            self.margin = 0

        if n is not None:
            is_integer(n, 'n')
        self.n = n

        super(Binomial, self).__init__(alpha=alpha, power=power,
                                       beta=beta, hypothesis='equality')

    def calculate(self):
        if self.n is None:
            self.calculate_n()
        elif self.power is None:
            self.calculate_power()
        elif self.alpha is None:
            self.calculate_alpha()

    def calculate_n(self):
        """ Perfrom the power calculation """
        if self.power is None:
            power = 0.8
        else:
            power = self.power

        if self.alpha is None:
            alpha = 0.5
        else:
            alpha = self.alpha

        random.seed(self._SEED)

        res = self._power_internals(self._minN, alpha)
        lag_lower = (self._minN, res[0])
        if lag_lower[1] > power:
            self.power = res[0]
            self.alpha = res[1]
            self.n = self._minN[0]
            return

        res = self._power_internals(self._maxN, alpha)
        lag_upper = (self._maxN, res[0])
        if lag_upper[1] < power:
            raise BaseException("N > " + str(self._maxN) +
                                ".  You should use large sample theory.")

        while True:
            delta_n = lag_upper[0] - lag_lower[0]
            test_n = math.floor(delta_n / 2) + lag_lower[0]
            test_power, test_alpha = self._power_internals(test_n, alpha)
            if test_power < power:
                # Look at upper half of what's left
                lag_lower = (test_n, test_power, test_alpha)
            else:
                # Otherwise, look at the lower half
                lag_upper = (test_n, test_power, test_alpha)
            if delta_n <= 1:
                # We've zeroed in.  Let's get out of the loop
                found_solution = True
                break

        if not found_solution:
            raise BaseException("N is greater than maximum N")

        if lag_lower[1] > power:
            self.n = lag_lower[0]
            self.power = lag_lower[1]
            self.alpha = lag_lower[2]
        else:
            self.n = lag_upper[0]
            self.power = lag_upper[1]
            self.alpha = lag_upper[2]
        self.beta = 1 - self.power

    def calculate_power(self):
        """ Perfrom the power calculation """
        if self.alpha is None:
            alpha = 0.05
        else:
            alpha = self.alpha

        random.seed(self._SEED)

        self.power, _ = self._power_internals(self.n, alpha)
        self.beta = 1 - self.power

    def calculate_alpha(self):
        """ Perfrom the power calculation """
        if self.power is None:
            power = 0.8
        else:
            power = self.power

        random.seed(self._SEED)

        res = random.binomial(self.n, self.p, self._N_SIMS)

        p_vals = []
        for x in res:
            p_val = stats.binom_test(x, n=self.n, p=self.p_0)
            p_vals.append(p_val)
        p_vals.sort()
        self.alpha = p_vals[int(self._N_SIMS * power)]

    def _power_internals(self, n, alpha):
        count = 0
        res = random.binomial(n, self.p, self._N_SIMS)
        p_vals = []
        for x in res:
            p_val = stats.binom_test(x, n=n, p=self.p_0)
            if p_val < alpha:
                count += 1
            p_vals.append(p_val)
        p_vals.sort()
        power = count / self._N_SIMS
        alpha = p_vals[int(self._N_SIMS * power) - 1]
        return power, alpha
