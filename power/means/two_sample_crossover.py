import math
from . import MeansBase
from power import is_numeric
from scipy.optimize import brenth
import scipy.stats as stats


class TwoSampleCrossover(MeansBase):
    """ Power and Sample Size calculations for two sample tests of means for
    a crossover study.

    The Two Sample Test of Means for Crossover Studies covers three hypothesis:
    equality, superiority, and equivalence.  Let :math:`\\mu_{1}` be the mean
    for treatment 1 and :math:`\\mu_{2}` be the mean for treamtment 2.

    The test for equality tests :math:`H_{0}: \\mu_{1}=\\mu_{2}` versus
    :math:`H_{1}: \\mu_{1} \\ne \\mu_{2}`.

    The test for superiority tests :math:`H_{0}: \\mu_{1} > \\mu_{2}` versus
    :math:`H_{1}: \\mu_{1} \\le \\mu_{2}`.

    The test for equivalence tests :math:`H_{0}: \\mu_{1} \\ne \\mu_{2}` versus
    :math:`H_{1}: \\mu_{1}=\\mu_{2}`.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        mu_1: :math:`\\mu_{1}` is the mean for treatment 1.
        mu_2: :math:`\\mu_{2}` is the the mean for treatment 2.
        sigma: :math:`\\sigma` is the standard deviation of the sample.
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
    """

    def __init__(self, n=None, mu_1=None, mu_2=None, stdev=None,
                 known_stdev=None, hypothesis=None, margin=None,
                 alpha=None, beta=None, power=None):

        is_numeric(mu_1, 'mu_1')
        is_numeric(mu_2, 'mu_2')

        if margin is not None:
            is_numeric(margin, 'margin')
        else:
            margin = 0

        epsilon = mu_1 - mu_2

        epsilon = float(mu_1) - float(mu_2)

        if hypothesis is 'superiority':
            epsilon = epsilon + float(margin)
        elif hypothesis is 'equivalence':
            # This should be margin - abs(epsilon), but the abs() is taken care
            # of when epsilon is set for generality purposes
            epsilon = float(margin) - abs(epsilon)

        epsilon *= math.sqrt(2)

        # Initialize the remaining arguments through the parent.
        super(TwoSampleCrossover, self).__init__(n=n,
                                                 epsilon=epsilon,
                                                 stdev=stdev,
                                                 known_stdev=known_stdev,
                                                 alpha=alpha,
                                                 beta=beta,
                                                 power=power,
                                                 hypothesis=hypothesis)

    @staticmethod
    def _calculate_power_unknown(n, alpha, theta,
                                 _alpha_adjustment, _beta_adjustment):
        """ Calculate power in the case that stdev is unknown.

        This is an internal static method only.  This overrides the
        superclasses method.
        """
        nu = 2*n - 2
        ncp = math.sqrt(n) * abs(theta)
        quantile = stats.t.ppf(1 - alpha / _alpha_adjustment, nu)
        power = 1 - _beta_adjustment * stats.nct.cdf(quantile, nu, ncp)
        return power

    def calculate(self):
        """ Performs the power calculation """
        if self.known_stdev:
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
        else:
            if self.n is None:
                self._set_default_alpha()
                self._set_default_power()
                res = brenth(lambda n:
                             self._calculate_power_unknown(n,
                                                           self.alpha,
                                                           self.theta,
                                                           self._alpha_adjustment,
                                                           self._beta_adjustment) - self.power,
                             a=2, b=1e7)
                self.n = math.ceil(res)
                self.power = self._calculate_power_unknown(self.n,
                                                           self.alpha,
                                                           self.theta,
                                                           self._alpha_adjustment,
                                                           self._beta_adjustment)
            elif self.power is None:
                self._set_default_alpha()
                self.power = self._calculate_power_unknown(self.n,
                                                           self.alpha,
                                                           self.theta,
                                                           self._alpha_adjustment,
                                                           self._beta_adjustment)
            elif self.alpha is None:
                n = self.n
                res = brenth(lambda alpha:
                             self._calculate_power_unknown(n,
                                                           alpha,
                                                           self.theta,
                                                           self._alpha_adjustment,
                                                           self._beta_adjustment) - self.power,
                             a=0, b=1)
                self.alpha = res
