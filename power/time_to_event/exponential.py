import math
from power.means import TwoSampleParallel
from power import (is_non_negative,
                            is_positive)
import scipy.stats as stats


class Exponential(TwoSampleParallel):
    """ Hypotheses that compares two exponential survival functions.

    Let :math:`\\delta = \\lambda_{c} - \\lambda_{t}`.  The exponential
    survival model allows you to test one of three hypotheses about
    :math:`\\delta`.

    Equality: The test of equality tests the null hypothesis that
        :math:`\\delta = 0` versus the alternative that
        :math:`\\delta \\ne 0`.
    Superiority and Non-Inferiority: The test of superiority (and the test
        of non-inferiority, which differs only in the direction of the
        comparison) tests the null hypothesis that
        :math:`\\delta \\le \\epsilon` versus the alternative that
        :math:`\\delta \\gt \\epsilon` where :math:`\\epsilon` is
        the margin above which two means are considered different.
    Equivalence: The test of equivalence tests the null hypothesis that
        :math:`|\\delta| \\ge \\epsilon` versus the alternative that
        :math:`|\\delta| \\lt \\epsilon` where :math:`\\epsilon` is
        the margin above which two means are considered different.

    Attributes:
        n_1: The control sample size.
        n_2: The treatment sample size.
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        control_hazard: The hazard rate, :math:`\lambda`, of the control group.
        treatment_hazard: The hazard rate, :math:`\lambda`, of the
            treatment group.
        trial_time: The total time the trial runs, :math:`T`.
        accrual_time: The time dedicated to accrual, :math:`T_{0}`.
        gamma: The patient entry parameter.  The default is 1.
        margin: (optional) The margin of the test, :math:`\\epsilon`.  The
            default is 0.
        hypothesis: (optional) An indication of the type of hypothesis.  It is
            equal to either 'equality', 'superiority' or 'equivalence'.  The
            default is 'equality'.
        ratio: (optional) The ratio between the treatment and control sample
            sizes.  If :math:`n_{1}` is the control sample size and
            :math:`n_{2}` is the treatment sample size, the ratio, :math:`k`,
            is defined as :math:`n_{1} = k n_{2}`.  The default value is 1.

    Notes:
        The follow up time is defined as :math:`T - T_{0}`.
    """

    def __init__(self, n_1=None, n_2=None, ratio=1,
                 alpha=None, power=None, beta=None, hypothesis=None,
                 margin=None, control_hazard=None, treatment_hazard=None,
                 gamma=None, trial_time=None, accrual_time=None):

        is_positive(accrual_time, 'Accrual Time')
        is_positive(trial_time, 'Trial Time')
        if gamma is None:
            gamma = 1
        else:
            is_non_negative(gamma, 'Gamma')

        g = gamma
        t = trial_time
        t_0 = accrual_time
        variance = []

        is_positive(control_hazard, 'Control Hazard')
        is_positive(treatment_hazard, 'Treatment Hazard')

        for hazard in [control_hazard, treatment_hazard]:
            if g == 0:
                EVNum = (math.exp(-hazard * t) - math.exp(-hazard * (t - t_0)))
                EVDenom = hazard * t_0
            else:
                EVNum = (g * math.exp(-1 * hazard * t) *
                         (1 - math.exp((hazard - g) * t_0)))
                EVDenom = (hazard - g) * (1 - math.exp(-g * t_0))
            EV = 1 + EVNum / EVDenom
            variance.append(hazard**2 / EV)

        self.stdev_control = math.sqrt(variance[0])
        self.stdev_treatment = math.sqrt(variance[1])

        # Initialize the remaining arguments through the parent.
        super(Exponential, self).__init__(n_1=n_1, n_2=n_2, ratio=ratio,
                                          mu_1=control_hazard,
                                          mu_2=treatment_hazard, stdev=1,  # dummy value
                                          known_stdev=True, alpha=alpha,
                                          beta=beta, power=power,
                                          margin=margin, hypothesis=hypothesis)

    def _calculate_n_known(self):
        """ Calculate n in the case that the standard deviation is known.

        This is an internal method only.
        """
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / self._alpha_adjustment)
        z_beta = distribution.ppf(1 - self.beta / self._beta_adjustment)

        n_2 = (z_alpha + z_beta)**2 / self.epsilon**2 * (self.stdev_control**2 / self.ratio + self.stdev_treatment**2)
        self.n_2 = math.ceil(n_2)
        self.n_1 = math.ceil(self.ratio * self.n_2)
        self.n = self.n_1 + self.n_2

    def _calculate_alpha_known(self):
        """ Calculate :math:`\\alpha` in the case that the standard deviation
        is known.

        This is an internal method only.
        """
        theta = (self.stdev_control**2 / self.ratio + self.stdev_treatment**2) / self.epsilon**2
        theta = 1 / math.sqrt(theta)
        distribution = stats.norm()
        z_beta = distribution.ppf(1 - self.beta / self._beta_adjustment)
        z_alpha = math.sqrt(self.n_2) * abs(theta) - z_beta

        self.alpha = (1 - distribution.cdf(z_alpha)) * self._alpha_adjustment

    def _calculate_power_known(self):
        """ Calculate power in the case that the standard deviation is known.

        This is an internal method only.
        """
        theta = (self.stdev_control**2 / self.ratio + self.stdev_treatment**2) / self.epsilon**2
        theta = 1 / math.sqrt(theta)
        distribution = stats.norm()
        z_alpha = distribution.ppf(1 - self.alpha / self._alpha_adjustment)
        z_beta = math.sqrt(self.n_2) * abs(theta) - z_alpha

        self.beta = (1 - stats.norm.cdf(z_beta)) * self._beta_adjustment
        self.power = 1 - self.beta

    def calculate(self):
        """ Performs the power calculation """
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

    def __repr__(self):
        """ The canonical representation of a TwoSampleParallel object
        """
        representation = "Alpha: " + str(self.alpha) + "\n" + \
                         "Power: " + str(self.power) + "\n" + \
                         "Sample Size (Group 1): " + str(self.n_1) + "\n" \
                         "Sample Size (Group 2): " + str(self.n_2) + "\n"
        return representation
