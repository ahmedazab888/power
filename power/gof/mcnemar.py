from power import (PowerBase,
                            is_in_0_1,
                            is_integer)
import math
import scipy.stats as stats


class McNemar(PowerBase):
    """ McNemar's Test for Categorical Shift

    McNemar's Test for Categorical Shift tests the hypothesis that:

    :math:`H_{0}: p_{10} = p_{01}` vs
    :math:`H_{0}: p_{10} \\ne p_{01}`

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        p_01: The value of :math:`p_{01}`
        p_10: The value of :math:`p_{10}`
    """
    def __init__(self, n=None, alpha=None, beta=None, power=None, p_01=None,
                 p_10=None):

        is_in_0_1(p_01, 'p_01 should be in [0, 1].')
        is_in_0_1(p_10, 'p_10 should be in [0, 1].')

        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        self.p_01 = p_01
        self.p_10 = p_10
        self.alpha_factor = math.sqrt(p_01 + p_10)
        self.beta_factor = math.sqrt(p_10 + p_01 - (p_01 - p_10)**2)

        # Set remaining variables.
        super(McNemar, self).__init__(alpha=alpha, beta=beta, power=power)

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

        n = ((z_alpha * self.alpha_factor + z_beta * self.beta_factor)**2 /
             (self.p_10 - self.p_01)**2)
        self.n = math.ceil(n)

    def _calculate_alpha_known(self):
        """ Calculate :math:`\\alpha`

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_beta = distribution.ppf(1 - self.beta)
        z_alpha = (math.sqrt(self.n) * abs(self.p_10 - self.p_01) -
                   z_beta * self.beta_factor) / self.alpha_factor

        self.alpha = (1 - distribution.cdf(z_alpha)) * 2.0

    def _calculate_power_known(self):
        """ Calculate power

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)
        z_beta = (math.sqrt(self.n) * abs(self.p_10 - self.p_01) -
                  z_alpha * self.alpha_factor) / self.beta_factor

        self.beta = (1 - stats.norm.cdf(z_beta))
        self.power = 1 - self.beta
