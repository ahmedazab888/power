from power.gof import GofBase
from power import (is_in_0_1,
                            is_integer)
import math
import scipy.optimize as optimize


class StuartMaxwell(GofBase):
    """ Hypotheses for Stuart-Maxwell's test for Categorical Shift.

    The Stuart-Maxwell Test for Categorical Shift compares the following
    hypothesis:

    :math:`H_{0}: p_{ij} = p_{ji}` for all :math:`i \\ne j` vs
    :math:`H_{0}: p_{ij} \\ne p_{ji}` for some :math:`j \\ne i`

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        p: The estimated probabilities as a list of lists.
    """
    def __init__(self, n=None, alpha=None, beta=None, power=None, p=None):
        # There should be a better way.
        # This loop both checks the values for p and caclualtes a value needed
        # for the calculation.
        self._denom = 0
        if isinstance(p, list):
            len_p = len(p)
            self.df = len_p * (len_p - 1) / 2.0
            self._adjustment = 0
            for i in range(len_p - 1):
                if isinstance(p, list):
                    for j in range(i, len_p):
                        is_in_0_1(p[i][j],
                                  'All values of p should be in [0, 1].')
                        is_in_0_1(p[j][i],
                                  'All values of p should be in [0, 1].')
                        self._denom += ((p[i][j] - p[j][i])**2 /
                                        (p[i][j] + p[j][i]))
                else:
                    raise ValueError("Each list in `p` must be a "
                                     "list numerics")
        else:
            raise ValueError("`p` must be a list of lists  of numerics")

        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        # Set remaining variables.
        super(StuartMaxwell, self).__init__(alpha=alpha,
                                            beta=beta,
                                            power=power)

    def calculate(self):
        df = self.df

        if self.n is None:
            if self.alpha is None:
                self.alpha = 0.05
            if self.power is None:
                power = 0.8
            else:
                power = self.power
            beta = 1 - power

            delta = optimize.brenth(lambda delta:
                                    self._beta(df, delta, self.alpha) - beta,
                                    a=0.00001, b=1e7)
            n = delta / self._denom
            self.n = math.ceil(n)
            self.beta = beta
            self.power = power
