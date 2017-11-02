import math
from . import MeansBase
from power import is_numeric
from scipy.optimize import brenth
import scipy.stats as stats


class TwoSampleParallel(MeansBase):
    """ Power and Sample Size calculations for two sample tests of means for
    a parallel study.

    The Two Sample Test of Means for Parallel Studies covers three hypothesis:
    equality, superiority, and equivalence.  Let :math:`\\mu_{1}` be the mean
    for group 1 and :math:`\\mu_{2}` be the mean for group 2.

    The test for equality tests :math:`H_{0}: \\mu_{1} = \\mu_{2}` versus
    :math:`H_{1}: \\mu_{1} \\ne \\mu_{2}`.

    The test for superiority tests :math:`H_{0}: \\mu_{1} > \\mu_{2}` versus
    :math:`H_{1}: \\mu_{1} \\le \\mu_{2}`.

    The test for equivalence tests :math:`H_{0}: \\mu_{1} \\ne \\mu_{2}` versus
    :math:`H_{1}: \\mu_{1} = \\mu_{2}`.

    Attributes:
        n_1: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta` for
            group 1.
        n_2: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta` for
            group 2.
        mu_1: :math:`\\mu` is the mean for group 1.
        mu_2: :math:`\\mu_{0}` is the the mean for group 2.
        stdev: :math:`\\sigma` is the standard deviation of the sample.
        known_stdev: A boolean indicator if the standard deviation.
        hypothesis: One of 'equality', 'superiority', or 'equvalence'.  Tests
            of 'non-inferiority' are the same as tests of 'superiority' with
            respect to the power calculation so choose 'superiority' for both.
        margin: This is the superiority or equivalence margin.
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        ratio: The ratio of n_1 to n_2.
    """

    def __init__(self, ratio=None, n_1=None, n_2=None,
                 mu_1=None, mu_2=None, stdev=None,
                 known_stdev=None, hypothesis=None, margin=None,
                 alpha=None, beta=None, power=None):
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

        self.n_1 = n_1
        self.n_2 = n_2
        self.ratio = float(ratio)

        stdev *= math.sqrt(1 + 1 / float(ratio))

        is_numeric(mu_1, 'mu_1')
        is_numeric(mu_2, 'mu_2')

        if margin is not None:
            is_numeric(margin, 'margin')

        epsilon = float(mu_1) - float(mu_2)

        if hypothesis is 'superiority':
            epsilon = epsilon + float(margin)
        elif hypothesis is 'equivalence':
            # This should be margin - abs(epsilon), but the abs() is taken care
            # of when epsilon is set for generality purposes
            epsilon = float(margin) - abs(epsilon)

        # Initialize the remaining arguments through the parent.
        super(TwoSampleParallel, self).__init__(n=n,
                                                epsilon=epsilon,
                                                stdev=stdev,
                                                known_stdev=known_stdev,
                                                alpha=alpha,
                                                beta=beta,
                                                power=power,
                                                hypothesis=hypothesis)

    # Means methods are being overridden because of the special treatment
    # required to accomidate an unbalanced design.
    @staticmethod
    def _calculate_power_unknown(n_2, alpha, theta, ratio,
                                 _alpha_adjustment, _beta_adjustment):
        """ Calculate power in the case that stdev is unknown.

        This is an internal static method only.  This overrides the
        superclasses method.
        """
        nu = (1 + ratio) * n_2 - 2
        ncp = math.sqrt(n_2) * abs(theta)
        quantile = stats.t.ppf(1 - alpha / _alpha_adjustment, nu)
        power = 1 - _beta_adjustment * stats.nct.cdf(quantile, nu, ncp)
        return power

    def _calculate_n_known(self):
        """ Calculate n in the case that the standard deviation is known.

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / self._alpha_adjustment)
        z_beta = distribution.ppf(1 - self.beta / self._beta_adjustment)

        n_2 = (z_alpha + z_beta)**2 / self.theta**2
        self.n_2 = math.ceil(n_2)
        self.n_1 = math.ceil(self.ratio * self.n_2)

    def _calculate_alpha_known(self):
        """ Calculate :math:`\\alpha` in the case that the standard deviation
        is known.

        This is an internal method only.
        """
        distribution = stats.norm()
        z_beta = distribution.ppf(1 - self.beta / self._beta_adjustment)
        z_alpha = math.sqrt(self.n_2) * abs(self.theta) - z_beta

        self.alpha = (1 - distribution.cdf(z_alpha)) * self._alpha_adjustment

    def _calculate_power_known(self):
        """ Calculate power in the case that the standard deviation is known.

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / self._alpha_adjustment)
        z_beta = math.sqrt(self.n_2) * abs(self.theta) - z_alpha

        self.beta = (1 - stats.norm.cdf(z_beta)) * self._beta_adjustment
        self.power = 1 - self.beta

    def calculate(self):
        """ Performs the power calculation """
        if self.known_stdev:
            if self.n is None:
                self._set_default_alpha()
                self._set_default_power()
                self._calculate_n_known()
                self._calculate_power_known()
                self.n = self.n_1 + self.n_2
            elif self.power is None:
                self._set_default_alpha()
                self._calculate_power_known()
            elif self.alpha is None:
                self._calculate_alpha_known()
        else:
            if self.n is None:
                self._set_default_alpha()
                self._set_default_power()
                alpha = self.alpha
                res = brenth(lambda n:
                             self._calculate_power_unknown(n,
                                                           alpha,
                                                           self.theta,
                                                           self.ratio,
                                                           self._alpha_adjustment,
                                                           self._beta_adjustment) - self.power,
                             a=2, b=1e7)
                self.n_2 = math.ceil(res)
                self.n_1 = math.ceil(self.n_2 * self.ratio)
                self.n = self.n_1 + self.n_2
                self.power = self._calculate_power_unknown(self.n_2,
                                                           self.alpha,
                                                           self.theta,
                                                           self.ratio,
                                                           self._alpha_adjustment,
                                                           self._beta_adjustment)
            elif self.power is None:
                self._set_default_alpha()
                self.power = self._calculate_power_unknown(self.n_2,
                                                           self.alpha,
                                                           self.theta,
                                                           self.ratio,
                                                           self._alpha_adjustment,
                                                           self._beta_adjustment)
            elif self.alpha is None:
                res = brenth(lambda alpha:
                             self._calculate_power_unknown(self.n_2,
                                                           alpha,
                                                           self.theta,
                                                           self.ratio,
                                                           self._alpha_adjustment,
                                                           self._beta_adjustment) - self.power,
                             a=0, b=1)
                self.alpha = res

    def __repr__(self):
        """ The canonical representation of a TwoSampleParallel object
        """
        representation = "Alpha: " + str(self.alpha) + "\n" + \
                         "Power: " + str(self.power) + "\n" + \
                         "Sample Size (Group 1): " + str(self.n_1) + "\n" \
                         "Sample Size (Group 2): " + str(self.n_2) + "\n"
        return representation
