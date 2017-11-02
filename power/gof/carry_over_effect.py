from power import (PowerBase,
                            is_in_0_1,
                            is_integer,
                            is_numeric,
                            is_positive)
import math
import scipy.stats as stats


class CarryOverEffect(PowerBase):
    """ Test for presence of a carry over effect with binary outcome

    A 2 by 2 crossover design's effects can be decomposed into treatment,
    period and crossover effects.  The hypothesis to test for the presence of
    a crossover effect is given by:

    :math:`H_{0}: \\gamma = 0` vs
    :math:`H_{0}: \\gamma \\ne 0`

    where :math:`\\gamma` is the sum of the carry over effects.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        p_11: The proportion with effect for treatment 1 in period 1
        p_12: The proportion with effect for treatment 1 in period 2
        p_21: The proportion with effect for treatment 2 in period 1
        p_22: The proportion with effect for treatment 2 in period 2
        gamma: The crossover effect (either gamma or the probabilities are
            required)
        stdev_1: The standard deviation within period 1
        stdev_2: The standard deviation within period 2
    """
    def __init__(self, n=None, alpha=None, beta=None, power=None,
                 p_11=None, p_12=None, p_21=None, p_22=None,
                 gamma=None, stdev_1=None, stdev_2=None):

        if gamma is None:
            is_in_0_1(p_11, 'p_11 should be in [0, 1].')
            is_in_0_1(p_12, 'p_12 should be in [0, 1].')
            is_in_0_1(p_21, 'p_21 should be in [0, 1].')
            is_in_0_1(p_22, 'p_22 should be in [0, 1].')

            gamma = (math.log(p_11 / (1 - p_11)) +
                     math.log(p_12 / (1 - p_12)) -
                     math.log(p_21 / (1 - p_21)) -
                     math.log(p_22 / (1 - p_22)))
        else:
            is_numeric(gamma, '`gamma` should be a number.')

        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        is_positive(stdev_1, 'stdev_1')
        self.stdev_1 = stdev_1

        is_positive(stdev_2, 'stdev_2')
        self.stdev_2 = stdev_2

        self.theta = gamma / math.sqrt(stdev_1**2 + stdev_2**2)

        # Set remaining variables.
        super(CarryOverEffect, self).__init__(alpha=alpha,
                                              beta=beta,
                                              power=power)

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
        """ Calculate n in the case that the standard deviation is known.

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / self._alpha_adjustment)
        z_beta = distribution.ppf(1 - self.beta / self._beta_adjustment)

        n = (z_alpha + z_beta)**2 / self.theta**2
        self.n = math.ceil(n)

    def _calculate_alpha_known(self):
        """ Calculate :math:`\\alpha` in the case that the standard deviation
        is known.

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_beta = distribution.ppf(1 - self.beta / self._beta_adjustment)
        z_alpha = math.sqrt(self.n) * self.theta - z_beta

        self.alpha = (1 - distribution.cdf(z_alpha)) * self._alpha_adjustment

    def _calculate_power_known(self):
        """ Calculate power in the case that the standard deviation is known.

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / self._alpha_adjustment)
        z_beta = math.sqrt(self.n) * self.theta - z_alpha

        self.beta = (1 - stats.norm.cdf(z_beta)) * self._beta_adjustment
        self.power = 1 - self.beta
