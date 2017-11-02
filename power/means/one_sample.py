import math
from power import is_numeric
from . import MeansBase
from scipy.optimize import brenth


class OneSample(MeansBase):
    """ Power and Sample Size calculations for one sample tests of means.

    The One Sample Test of Means covers three hypothesis: equality,
    superiority, and equivalence.  Let :math:`\\mu` be the unknown mean and
    :math:`\\mu_{0}` be the mean that it is being tested against.

    The test for equality tests :math:`H_{0}: \\mu = \\mu_{0}` versus
    :math:`H_{1}: \\mu \\ne \\mu_{0}`.

    The test for superiority tests :math:`H_{0}: \\mu > \\mu_{0}` versus
    :math:`H_{1}: \\mu \\le \\mu_{0}`.

    The test for equivalence tests :math:`H_{0}: \\mu \\ne \\mu_{0}` versus
    :math:`H_{1}: \\mu = \\mu_{0}`.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        mu: :math:`\\mu` is the mean under the null hypothesis.
        mu_0: :math:`\\mu_{0}` is the mean under the alternative hypothesis.
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
    """

    def __init__(self, n=None, mu=None, mu_0=None, stdev=None,
                 known_stdev=None, hypothesis=None, margin=None,
                 alpha=None, beta=None, power=None):

        is_numeric(mu, 'mu')

        is_numeric(mu_0, 'mu_0')

        if margin is not None:
            is_numeric(margin, 'margin')

        # Ensure that the values are floats
        epsilon = abs(float(mu) - float(mu_0))

        if hypothesis is 'superiority':
            epsilon = epsilon - float(margin)
        elif hypothesis is 'equivalence':
            epsilon = float(margin) - abs(epsilon)

        # Initialize the remaining arguments through the parent.
        super(OneSample, self).__init__(n=n, epsilon=epsilon, stdev=stdev,
                                        known_stdev=known_stdev, alpha=alpha,
                                        beta=beta, power=power,
                                        hypothesis=hypothesis)

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
