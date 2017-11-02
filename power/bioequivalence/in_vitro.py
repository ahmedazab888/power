from power import (PowerBase,
                   is_in_0_1,
                   is_integer,
                   is_numeric,
                   is_positive)
import scipy.stats as stats
import math

MAX_ITERATIONS = 1000


class InVitro(PowerBase):
    """ Sample Size calculations for tests of individual bioequivalence.

    This class calculates the sample size required for testing InVitro
    Bioequivalence using a 2 x 4 crossover design.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        delta: :math:`\\delta` is the difference in bioequivalence
        stdev_wr: :math:`\\sigma_{WR}`
        stdev_wt: :math:`\\sigma_{WT}`
        stdev_br: :math:`\\sigma_{BR}`
        stdev_bt: :math:`\\sigma_{BT}`
        theta_BE: :math:`\\theta_{BE}` is the BE limit specificed from the
            FDA (1.74).
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
    """

    def __init__(self, delta=None, stdev_wr=None, stdev_wt=None,
                 stdev_br=None, stdev_bt=None, theta_BE=None, m_plus=None,
                 alpha=None, power=None, beta=None):
        is_numeric(delta, 'delta')
        self.delta = delta

        is_positive(stdev_wr, 'stdev_wr')
        self.stdev_wr = stdev_wr
        is_positive(stdev_wt, 'stdev_wt')
        self.stdev_wt = stdev_wt
        is_positive(stdev_bt, 'stdev_bt')
        self.stdev_bt = stdev_bt
        is_positive(stdev_br, 'stdev')
        self.stdev_br = stdev_br

        if theta_BE is None:
            theta_BE = 1
        else:
            is_positive(theta_BE, 'theta_BE')
        self.theta_BE = theta_BE

        if m_plus is None:
            m_plus = MAX_ITERATIONS
        else:
            is_integer(m_plus, 'm_plus')
        self.m_plus = m_plus

        # Initialize the remaining arguments through the parent.
        super(InVitro, self).__init__(hypothesis="equivalence",
                                      alpha=alpha, beta=beta, power=power)

    def _calculate_gamma(self):
        stdev_tt = math.sqrt(self.stdev_wt**2 + self.stdev_bt**2)
        stdev_tr = math.sqrt(self.stdev_wr**2 + self.stdev_br**2)
        gamma = self.delta**2 + stdev_tt**2 - (1 + self.theta_BE) * stdev_tr**2
        return gamma

    def _calculate_u(self, m, n, cut):
        dist = stats.norm()
        tmp_1 = ((m - 1) / stats.chi2.ppf(1 - cut, df=m - 1) - 1)**2
        tmp_2 = (n * (m - 1) / stats.chi2.ppf(1 - cut, df=n * (m - 1)) - 1)**2
        tmp_3 = ((m - 1) / stats.chi2.ppf(cut, df=m - 1) - 1)**2
        tmp_4 = (n * (m - 1) / stats.chi2.ppf(cut, df=n * (m - 1)) - 1)**2
        c = 1
        U = ((abs(self.delta) +
              dist.ppf(cut) *
              math.sqrt((self.stdev_bt**2 + self.stdev_br**2) / m))**2 -
             self.delta**2)**2
        U += self.stdev_bt**4 * tmp_1
        U += (1 - 1 / n)**2 * self.stdev_wt**4 * tmp_2
        U += (1 + self.theta_BE)**2 * self.stdev_br**4 * tmp_3
        U += ((1 + c * self.theta_BE)**2 * (1 - 1 / n)**2 *
              self.stdev_wr**4 * tmp_4)
        return U

    def _calculate_sigma(self, a, b):
        sigma = (self.stdev_d**2 + a * self.stdev_wt**2 +
                 b * self.stdev_wr**2)
        return sigma

    def calculate(self):
        gamma = self._calculate_gamma()
        # Step 1 is the first iteration of the inner loop.
        # Step 2 is the inner loop.
        # Step 3 is the outer loop
        for n in range(1, MAX_ITERATIONS):
            for m in range(30, self.m_plus):
                U = math.sqrt(self._calculate_u(m, n, 0.05))
                U_beta = math.sqrt(self._calculate_u(m, n, self.power))
                print(str(n) + ', ' + str(m))
                print(str(gamma) + ', ' + str(U) + ', ' + str(U_beta))
                if gamma + U + U_beta <= 0:
                    self.n = n
                    self.m = m
                    return
