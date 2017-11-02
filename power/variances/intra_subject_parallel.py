import math
from power.variances import VarianceBase
import scipy.stats as stats
from scipy.optimize import brenth


class IntraSubjectParallel(VarianceBase):
    """ Tests for Intrasubject Variablility using a Crossover study with
    :math:`m` replicates.

    The test for equality tests:
    :math:`H_{0}: \\frac{\\sigma_{WT}}{\\sigma_{WR}} = 1` versus
    :math:`H_{1}: \\frac{\\sigma_{WT}}{\\sigma_{WR}} \\ne 1`.

    The test for superiority tests:
    :math:`H_{0}: \\frac{\\sigma_{WT}}{\\sigma_{WR}} \\ge \\delta` versus
    :math:`H_{1}: \\frac{\\sigma_{WT}}{\\sigma_{WR}} < \\delta`.

    The test for similarity tests:
    :math:`H_{1}: \\frac{\\sigma_{WT}}{\\sigma_{WR}} \\ge \\delta` or
    :math:`\\frac{\\sigma_{WT}}{\\sigma_{WR}} \\ge 1/\\delta`
    versus
    :math:`H_{1}: 1/\\delta \\le \\frac{\\sigma_{WT}}{\\sigma_{WR}} < \\delta`.

    Attributes:
        n: The number of subjects required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        m: The number of replicates per subject.
        stdev_wt: The variance within treatment.
        stdev_wr: The variance within replicates.
        similarity_limit: the percent of variances that is clinically
            significant.

        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
    """

    def __init__(self, n=None, m=None, stdev_wt=None, stdev_wr=None,
                 similarity_limit=None, hypothesis=None,
                 alpha=None, beta=None, power=None):

        super(IntraSubjectParallel, self).__init__(n=n,
                                                   m=m,
                                                   stdev_1=stdev_wr,
                                                   stdev_2=stdev_wt,
                                                   similarity_limit=similarity_limit,
                                                   hypothesis=hypothesis,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   power=power)

    @staticmethod
    def _calculate_power_unknown(n, m, sigma_ratio, alpha, beta):
        """
        """
        df = n * (m - 1)
        ratioLeft = sigma_ratio**2

        ratioRight = (stats.f.ppf(beta, df, df) /
                      stats.f.ppf(1 - alpha, df, df))
        res = ratioLeft - ratioRight
        return res

    def calculate(self):
        if self.n is None:
            self._set_default_alpha()
            self._set_default_power()
            res = brenth(lambda n:
                         self._calculate_power_unknown(n,
                                                       self.m,
                                                       self.sigma_ratio,
                                                       self._alpha,
                                                       self._beta),
                         a=2, b=1e7)
            self.n = math.ceil(res)
            res = brenth(lambda beta:
                         self._calculate_power_unknown(self.n,
                                                       self.m,
                                                       self.sigma_ratio,
                                                       self._alpha,
                                                       beta),
                         a=0, b=1)
            self._beta = res
            self.update_beta()
        elif self.power is None:
            self._set_default_alpha()
            res = brenth(lambda beta:
                         self._calculate_power_unknown(self.n,
                                                       self.m,
                                                       self.sigma_ratio,
                                                       self._alpha,
                                                       beta),
                         a=1e-7, b=1 - 1e-7)
            self._beta = res
            self.update_beta()
        elif self.alpha is None:
            res = brenth(lambda alpha:
                         self._calculate_power_unknown(self.n,
                                                       self.m,
                                                       self.sigma_ratio,
                                                       alpha,
                                                       self._beta),
                         a=1e-7, b=1 - 1e-7)
            self._alpha = res
            self.update_alpha()
