from power.gof import GofBase
from power import (is_in_0_1,
                            is_integer)
import math
import scipy.stats as stats


class CMH(GofBase):
    """ Cochran-Mantel-Haenszel Test for Independance (Multiple Strata)

    Cochran-Mantel-Haenszel Independance Test uses a Chi Squared test to test
    for the independance in multiple strata.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        p: A list of lists of lists of observed probabilities.  The depth of
            the inception is the number of strata
        pi: The proportion of subjects in each strata
    """
    def __init__(self, n=None, alpha=None, beta=None, power=None, p=None,
                 pi=None):
        self._check_list(p, 'p')

        if pi is None:
            pi = [1 / float(len(p))] * len(p)
        else:
            self._check_list(pi, 'pi')

        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        num = 0
        denom = 0
        for stratum in range(len(p)):
            rowsums = [sum(p[stratum][0]), sum(p[stratum][1])]
            colsums = [p[stratum][0][0] + p[stratum][1][0],
                       p[stratum][0][1] + p[stratum][1][1]]
            num += pi[stratum] * (p[stratum][0][0] - rowsums[0] * colsums[0])
            denom += (pi[stratum] * rowsums[0] * rowsums[1] *
                      colsums[0] * colsums[1])

        self.delta = num / math.sqrt(denom)
        self.n = n

        # Set remaining variables.
        super(CMH, self).__init__(alpha=alpha, beta=beta, power=power)

    def calculate(self):
        if self.n is None:
            self._set_default_alpha()
            self._set_default_power()
            self._calculate_n_known()
            self._calculate_power_known()
        elif self.power is None:
            self._set_default_alpha()
            self._calculate_power_known()
        elif self.alpha is None:
            self._calculate_alpha_known()

    def _calculate_n_known(self):
        """ Calculate n

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)
        z_beta = distribution.ppf(1 - self.beta)

        n = (z_alpha + z_beta)**2 / self.delta**2
        self.n = math.ceil(n)

    def _calculate_alpha_known(self):
        """ Calculate :math:`\\alpha`

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_beta = distribution.ppf(1 - self.beta)
        z_alpha = math.sqrt(self.n) * self.delta - z_beta

        self.alpha = (1 - distribution.cdf(z_alpha)) * 2.0

    def _calculate_power_known(self):
        """ Calculate power

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)
        z_beta = math.sqrt(self.n) * self.delta - z_alpha

        self.beta = (1 - stats.norm.cdf(z_beta))
        self.power = 1 - self.beta

    def _check_list(self, lst, label):
        """ Recursively check the lists in lst """
        if isinstance(lst, list):
            for values in lst:
                if isinstance(values, list):
                    self._check_list(values, label=label)
                else:
                    is_in_0_1(values,
                              ('All values of ' + label +
                               'p should be in (0, 1).'))
        else:
            raise ValueError("`p` must be a list of lists of numerics")
