from power import (PowerBase,
                            is_in_0_1)
import scipy.stats as stats
import math


class TwoSample(PowerBase):
    """
    """
    def __init__(self, n_1=None, n_2=None, ratio=None, alpha=None, beta=None,
                 power=None, p_1=None, p_2=None, p_3=None):
        # n is only used to help with control flow
        if ratio is None:
            if n_1 is None:
                ratio = 1
                if n_2 is None:
                    n = None
                else:
                    n_1 = n_2
                    n = n_1 + n_2
            else:
                if n_2 is None:
                    ratio = 1
                    n_2 = n_1
                    n = n_1 + n_2
                else:
                    n = n_1 + n_2
                    ratio = n_1 / float(n_2)
        else:
            if n_1 is None:
                if n_2 is None:
                    n = None
                else:
                    n_1 = math.ceil(ratio * n_2)
                    n = n_1 + n_2
            else:
                if n_2 is None:
                    n_2 = math.ceil(n_1 / ratio)
                    n = n_1 + n_2
                else:
                    n = n_1 + n_2
        self.n = n
        self.n_1 = n_1
        self.n_2 = n_2
        self.ratio = float(ratio)

        is_in_0_1(p_1, 'p_1 should be in [0, 1].')
        is_in_0_1(p_2, 'p_2 should be in [0, 1].')
        is_in_0_1(p_3, 'p_3 should be in [0, 1].')

        self.p_1 = p_1
        self.p_2 = p_2
        self.p_3 = p_3

        # Initialize the remaining arguments through the parent.
        super(TwoSample, self).__init__(alpha=alpha, power=power,
                                        beta=beta, hypothesis=None)

    def _calculate_n(self):
        """ Calculate n

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)
        z_beta = distribution.ppf(1 - self.beta)

        alpha_factor = math.sqrt(self.ratio * (self.ratio + 1) / 12)
        beta_factor = math.sqrt(self.ratio**2 * (self.p_2 - self.p_1**2) +
                                self.ratio * (self.p_3 - self.p_1**2))
        n_factor = self.ratio * (0.5 - self.p_1)
        n = (z_alpha * alpha_factor + z_beta * beta_factor)**2 / n_factor**2

        self.n_2 = math.ceil(n)
        self.n_1 = math.ceil(self.ratio * self.n_2)
        self.n = self.n_1 + self.n_2

    def _calculate_alpha(self):
        """ Calculate :math:`\\alpha`

        This is an internal method only.
        """
        distribution = stats.norm()
        z_beta = distribution.ppf(1 - self.beta)

        alpha_factor = math.sqrt(self.ratio * (self.ratio + 1) / 12)
        beta_factor = math.sqrt(self.ratio**2 * (self.p_2 - self.p_1**2) +
                                self.ratio * (self.p_3 - self.p_1**2))
        n_factor = self.ratio * (0.5 - self.p_1)

        z_alpha = math.sqrt(self.n_2) * abs(n_factor) - z_beta * beta_factor
        z_alpha = z_alpha / alpha_factor

        self.alpha = (1 - distribution.cdf(z_alpha)) * 2.0

    def _calculate_power(self):
        """ Calculate power

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / 2.0)

        alpha_factor = math.sqrt(self.ratio * (self.ratio + 1) / 12)
        beta_factor = math.sqrt(self.ratio**2 * (self.p_2 - self.p_1**2) +
                                self.ratio * (self.p_3 - self.p_1**2))
        n_factor = self.ratio * (0.5 - self.p_1)

        z_beta = math.sqrt(self.n_2) * abs(n_factor) - z_alpha * alpha_factor
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

    def __repr__(self):
        """ The canonical representation of a TwoSample object
        """
        representation = "Alpha: " + str(self.alpha) + "\n" + \
                         "Power: " + str(self.power) + "\n" + \
                         "Sample Size (Group 1): " + str(self.n_1) + "\n" \
                         "Sample Size (Group 2): " + str(self.n_2) + "\n"
        return representation
