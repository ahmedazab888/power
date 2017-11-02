from power import (PowerBase,
                            is_in_0_1,
                            is_integer)
import scipy.stats as stats
import math


class Independance(PowerBase):
    """
    """
    def __init__(self, n=None, alpha=None, beta=None, power=None,
                 p_1=None, p_2=None):
        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        is_in_0_1(p_1, 'p_1 should be in [0, 1].')
        is_in_0_1(p_2, 'p_2 should be in [0, 1].')

        self.p_1 = p_1
        self.p_2 = p_2

        # Initialize the remaining arguments through the parent.
        super(Independance, self).__init__(alpha=alpha, power=power,
                                           beta=beta, hypothesis=None)

    def _calculate_n(self):
        """ Calculate n

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)
        z_beta = distribution.ppf(1 - self.beta)

        alpha_factor = 1 / 3.0
        beta_factor = math.sqrt(2 * self.p_2 - 1 - (2 * self.p_1 - 1)**2)
        n_factor = (2 * self.p_1 - 1) / 2
        n = (z_alpha * alpha_factor + z_beta * beta_factor)**2 / n_factor**2

        self.n = math.ceil(n)

    def _calculate_alpha(self):
        """ Calculate :math:`\\alpha`

        This is an internal method only.
        """
        distribution = stats.norm()
        z_beta = distribution.ppf(1 - self.beta)

        alpha_factor = 1 / 3.0
        beta_factor = math.sqrt(2 * self.p_2 - 1 - (2 * self.p_1 - 1)**2)
        n_factor = (2 * self.p_1 - 1) / 2

        z_alpha = math.sqrt(self.n) * abs(n_factor) - z_beta * beta_factor
        z_alpha = z_alpha / alpha_factor

        self.alpha = (1 - distribution.cdf(z_alpha)) * 2.0

    def _calculate_power(self):
        """ Calculate power

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)

        alpha_factor = 1 / 3.0
        beta_factor = math.sqrt(2 * self.p_2 - 1 - (2 * self.p_1 - 1)**2)
        n_factor = (2 * self.p_1 - 1) / 2

        z_beta = math.sqrt(self.n) * abs(n_factor) - z_alpha * alpha_factor
        z_beta = z_beta / beta_factor

        self.beta = (1 - stats.norm.cdf(z_beta))
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
