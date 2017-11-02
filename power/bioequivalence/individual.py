from power import (PowerBase,
                   is_in_0_1,
                   is_numeric,
                   is_positive)
import scipy.stats as stats
import math

MAX_ITERATIONS = 1000


class Individual(PowerBase):
    """ Sample Size calculations for tests of individual bioequivalence.

    This class calculates the sample size required for testing Individual
    Bioequivalence using a 2 x 4 crossover design.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        delta: :math:`\\delta` is the individual bioequivalence
        stdev_wr: :math:`\\sigma_{WR}`
        stdev_wt: :math:`\\sigma_{WT}`
        stdev_d: :math:`\\sigma_{D}`
        theta_IBE: :math:`\\theta_{IBE}` is the IBE limit specificed from the
            FDA (1.74).
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
    """

    def __init__(self, delta=None, stdev_wr=None, stdev_wt=None,
                 stdev_br=None, stdev_bt=None, rho=None,
                 theta_IBE=None, alpha=None, power=None,
                 beta=None):
        is_numeric(delta, 'delta')
        self.delta = delta

        is_positive(stdev_wr, 'stdev_wr')
        self.stdev_wr = stdev_wr
        is_positive(stdev_wt, 'stdev_wt')
        self.stdev_wt = stdev_wt
        is_positive(stdev_bt, 'stdev_br')
        self.stdev_br = stdev_br
        is_positive(stdev_bt, 'stdev_bt')
        self.stdev_bt = stdev_bt
        is_in_0_1(rho, 'rho')
        self.rho = rho

        var_d = self.stdev_bt**2 + self.stdev_br**2 - 2 * self.rho * self.stdev_bt**2 * self.stdev_bt**2
        self.stdev_d = math.sqrt(var_d)
        if theta_IBE is None:
            theta_IBE = 1.74
        else:
            is_positive(theta_IBE, 'theta_IBE')
        self.theta_IBE = theta_IBE

        # Initialize the remaining arguments through the parent.
        super(Individual, self).__init__(hypothesis="equivalence",
                                         alpha=alpha, beta=beta, power=power)

    def _calculate_gamma(self):
        gamma = self.delta**2 + self.stdev_d**2 + self.stdev_wt**2
        gamma -= (self.theta_IBE + 1) * self.stdev_wr**2
        return gamma

    def _calculate_u(self, n, cut):
        nu = 2 * n - 1
        sigma = self._calculate_sigma(0.5, 0.5)
        tmp_1 = ((nu - 1) / stats.chi2.ppf(1 - cut, df=nu - 1) - 1)**2
        tmp_2 = ((nu - 1) / stats.chi2.ppf(cut, df=nu - 1) - 1)**2
        U = (abs(self.delta) +
             stats.t.ppf(cut, df=nu) * sigma * math.sqrt(2 / n) / 2 -
             self.delta**2)**2
        U += sigma**4 * tmp_1
        U += 0.25 * self.stdev_wt**4 * tmp_1
        U += (1.5 + self.theta_IBE)**2 * self.stdev_wr**4 * tmp_2
        return U

    def _calculate_sigma(self, a, b):
        sigma = (self.stdev_d**2 + a * self.stdev_wt**2 +
                 b * self.stdev_wr**2)
        return sigma

    def calculate(self):
        gamma = self._calculate_gamma()
        self.n = None
        found_solution = False
        for i in range(2, MAX_ITERATIONS):
            bound = gamma + math.sqrt(self._calculate_u(i, 0.05))
            bound += math.sqrt(self._calculate_u(i, self.power))
            if bound < 0:
                self.n = i
                found_solution = True
                break
        if not found_solution:
            raise BaseException("N > " + str(MAX_ITERATIONS))
