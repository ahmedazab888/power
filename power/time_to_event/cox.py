from power import (is_in_0_1,
                            is_positive)
from power.means import OneSample
import math


class Cox(OneSample):
    """ Hypotheses for Cox's Proportional Hazard Model.

    Let :math:`b` be the hazard ratio of two survival curves.  Cox's
    proportional hazard model contains the following hypotheses about
    :math:`b`.

    Equality: The test of equality tests the null hypothesis that
        :math:`b = 0` versus the alternative that
        :math:`b \\ne 0`.
    Superiority and Non-Inferiority: The test of superiority (and the test
        of non-inferiority, which differs only in the direction of the
        comparison) tests the null hypothesis that
        :math:`b \\le \\delta` versus the alternative that
        :math:`b \\gt \\delta` where :math:`\\delta` is
        the margin above which two means are considered different.
    Equivalence: The test of equivalence tests the null hypothesis that
        :math:`b \\ge \\delta` versus the alternative that
        :math:`b \\lt \\delta` where :math:`\\delta` is
        the margin above which two means are considered different.

    Attributes:
        n: The sample size for the calculation
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        hypothesis: One of 'equality', 'superiority', or 'equvalence'.  Tests
            of 'non-inferiority' are the same as tests of 'superiority' with
            respect to the power calculation so choose 'superiority' for both.
        margin: This is the superiority or equivalence margin.
        hazard_ratio: The ratio of the hazards of the two survival curves,
            :math:`b`.
        proportion_visible: (optional) The proportion of events that can be
            detected.  `proportion_visible` must be in (0, 1]. The default
            is 1.
        margin: (optional) The margin of the test, :math:`\\delta`.  The
            default is 0.
        hypothesis: (optional) An indication of the type of hypothesis.  It is
            equal to either 'equality', 'superiority' or 'equivalence'.  The
            default is 'equality'.
        ratio: (optional) The ratio between the treatment and control sample
            sizes.  If :math:`n_{1}` is the control sample size and
            :math:`n_{2}` is the treatment sample size, the ratio, :math:`k`,
            is defined as :math:`n_{1} = k n_{2}`.  The default value is 1.
    """

    def __init__(self, n=None, alpha=None, power=None, beta=None,
                 hypothesis=None, margin=None, hazard_ratio=None,
                 proportion_visible=None, control_proportion=None,
                 treatment_proportion=None):
        is_positive(hazard_ratio, 'Hazard Ratio')
        is_in_0_1(treatment_proportion, 'Treatment Proportion')
        is_in_0_1(control_proportion, 'Control Proportion')
        is_in_0_1(proportion_visible, 'Proportion Visible')

        epsilon = math.log(hazard_ratio)
        stdev = treatment_proportion * control_proportion * proportion_visible
        stdev = 1 / math.sqrt(stdev)

        super(Cox, self).__init__(n=n, mu=epsilon, mu_0=0,
                                  stdev=stdev, known_stdev=True, alpha=alpha,
                                  beta=beta, power=power, margin=margin,
                                  hypothesis=hypothesis)
