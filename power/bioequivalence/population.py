from power import (PowerBase,
                            is_in_0_1,
                            is_numeric,
                            is_positive)
import scipy.stats as stats
import math


class Population(PowerBase):
    """ Sample Size calculations for tests of population bioequivalence.

    This class calculates the sample size needed for testing Population
    Bioequivalence using a 2 x 2 crossover design.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        delta: :math:`\\delta` is the mean difference in the AUCs.
        l: :math:`\\lambda`
        stdev_11: :math:`\\sigma_{1,1}`
        stdev_tt: :math:`\\sigma_{TT}`
        stdev_tr: :math:`\\sigma_{TR}`
        stdev_bt: :math:`\\sigma_{BT}`
        stdev_br: :math:`\\sigma_{BR}`
        rho: :math:`\\rho`
        theta_PBE: :math:`\\theta_{PBE}` is the PBE limit specificed from the
            FDA (1.74).
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
    """

    def __init__(self, delta=None, l=None, stdev_11=None, stdev_tt=None,
                 stdev_tr=None, stdev_bt=None, stdev_br=None, rho=None,
                 theta_PBE=None, alpha=None, power=None, beta=None):

        if delta is None:
            delta = 0
        else:
            is_numeric(delta, 'delta')
        self.delta = delta

        is_numeric(l, 'l')
        self.l = l

        is_positive(stdev_11, 'stdev_11')
        self.stdev_11 = stdev_11
        is_positive(stdev_tt, 'stdev_tt')
        self.stdev_tt = stdev_tt
        is_positive(stdev_tr, 'stdev_tr')
        self.stdev_tr = stdev_tr
        is_positive(stdev_bt, 'stdev_bt')
        self.stdev_bt = stdev_bt
        is_positive(stdev_br, 'stdev')
        self.stdev_br = stdev_br

        is_in_0_1(rho, 'rho')
        self.rho = rho

        if theta_PBE is None:
            theta_PBE = 1.74
        else:
            is_positive(theta_PBE, 'theta_PBE')
        self.theta_PBE = theta_PBE

        # Initialize the remaining arguments through the parent.
        super(Population, self).__init__(hypothesis="equivalence",
                                         alpha=alpha, beta=beta, power=power)

    def calculate(self):
        dist = stats.norm()
        coef = (2 * self.delta**2 * self.stdev_11**2 +
                self.stdev_tt**4 +
                (1 + self.theta_PBE)**2 * self.stdev_tr**4 -
                2 * (1 + self.theta_PBE) *
                self.rho**2*self.stdev_bt**2 * self.stdev_br**2)
        n = (dist.ppf(1 - self.alpha) + dist.ppf(1 - self.beta))**2
        n *= coef / self.l**2
        self.n = math.ceil(n)
