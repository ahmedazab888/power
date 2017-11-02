import math
from power import (PowerBase,
                            is_integer,
                            is_numeric,
                            is_positive,
                            is_boolean)
import scipy.stats as stats


class MeansBase(PowerBase):
    """ The base for means specific hypothesis objects.

    The Means class is an class that serves as a base for other hypotheses
    involving means.  Most of the hypotheses on means can be unified by the
    hypothesis :math:`H_{0}: \\epsilon = 0` versus
    :math`H_{1}: \\epsilon \\ne 0`.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        epsilon: :math:`\\epsilon` is the difference between the means.
        stdev: :math:`\\sigma` is the standard deviation of the sample.
        known_stdev: (optional) A boolean indicator if the standard deviation.
            If not provided, the default value is True.
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
    """

    def __init__(self, n=None, epsilon=None, stdev=None, known_stdev=None,
                 alpha=None, beta=None, power=None, hypothesis=None):

        if n is not None:
            is_integer(n, 'N')
            self.n = n
        else:
            self.n = None

        is_numeric(epsilon, 'epsilon')
        self.epsilon = epsilon

        is_positive(stdev, 'stdev')
        self.stdev = stdev

        self.theta = self.epsilon / self.stdev

        if known_stdev is not None:
            is_boolean(known_stdev, 'known_stdev')
        else:
            known_stdev = True
        self.known_stdev = known_stdev

        # Initialize the remaining arguments through the parent.
        super(MeansBase, self).__init__(alpha=alpha, power=power,
                                        beta=beta, hypothesis=hypothesis)

    @staticmethod
    def _calculate_power_unknown(n, alpha, theta,
                                 _alpha_adjustment, _beta_adjustment):
        """ Calculate power in the case that stdev is unknown.

        This is an internal static method only.
        """
        nu = n - 1
        ncp = math.sqrt(n) * abs(theta)
        quantile = stats.t.ppf(1 - alpha / _alpha_adjustment, nu)
        power = (1 - stats.nct.cdf(quantile, nu, ncp) +
                 stats.nct.cdf(-1 * quantile, nu, ncp))
        return power

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
        z_alpha = math.sqrt(self.n) * abs(self.theta) - z_beta

        self.alpha = (1 - distribution.cdf(z_alpha)) * self._alpha_adjustment

    def _calculate_power_known(self):
        """ Calculate power in the case that the standard deviation is known.

        This is an internal static method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / self._alpha_adjustment)
        z_beta = math.sqrt(self.n) * abs(self.theta) - z_alpha

        self.beta = (1 - stats.norm.cdf(z_beta)) * self._beta_adjustment
        self.power = 1 - self.beta
