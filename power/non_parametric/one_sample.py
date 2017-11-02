from power import (PowerBase,
                            is_in_0_1,
                            is_integer)
import scipy.stats as stats
import math


class OneSample(PowerBase):
    """
    """
    def __init__(self, n=None, alpha=None, beta=None, power=None,
                 p_2=None, p_3=None, p_4=None):
        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        is_in_0_1(p_2, 'p_2 should be in [0, 1].')
        is_in_0_1(p_3, 'p_3 should be in [0, 1].')
        is_in_0_1(p_4, 'p_4 should be in [0, 1].')

        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4

        # Initialize the remaining arguments through the parent.
        super(OneSample, self).__init__(alpha=alpha, power=power,
                                        beta=beta, hypothesis=None)

    def _calculate_n(self):
        """ Calculate n

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)
        z_beta = distribution.ppf(1 - self.beta)

        n = ((z_alpha / math.sqrt(12) + z_beta *
              math.sqrt(self.p_3 + 4 * self.p_4 - 4 * self.p_2**2))**2 /
             (0.25 - self.p_2)**2)
        self.n = math.ceil(n)

    def _calculate_alpha(self):
        """ Calculate :math:`\\alpha`

        This is an internal method only.
        """
        distribution = stats.norm()
        z_beta = distribution.ppf(1 - self.beta)

        z_alpha = self.n * (0.25 - self.p_2)**2
        z_alpha = math.sqrt(z_alpha)
        z_alpha -= z_beta * (math.sqrt(self.p_3 + 4 *
                             self.p_4 - 4 * self.p_2**2))
        z_alpha *= math.sqrt(12)

        self.alpha = (1 - distribution.cdf(z_alpha)) * 2.0

    def _calculate_power(self):
        """ Calculate power

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)
        num = math.sqrt(self.n) * (0.25 - self.p_2) + (z_alpha / math.sqrt(12))
        denom = (math.sqrt(self.p_3 + 4 * self.p_4 - 4 * self.p_2**2))
        z_beta = num / denom

        self.beta = (1 - stats.norm.cdf(-z_beta))
        self.power = 1 - self.beta

    def calculate(self):
        if self.n is None:
            self._set_default_alpha()
            self._set_default_power()
            self._calculate_n()
            self._calculate_power()
        elif self.power is None:
            self._set_default_alpha()
            self._calculate_power()
        elif self.alpha is None:
            self._calculate_alpha()
