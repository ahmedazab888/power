from power import PowerBase
import scipy.stats as stats


class GofBase(PowerBase):
    """ Hypotheses that tests a goodness of fit.  This is a base for other
    classes that implement goodness of fit tests.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
    """

    def __init__(self, alpha=None, beta=None, power=None):
        # Error handling is handled at the Hypothesis level.
        super(GofBase, self).__init__(alpha=alpha, beta=beta, power=power)

    @staticmethod
    def _beta(df, delta, alpha):
        q = stats.chi2.ppf(1 - alpha, df=df)
        beta = stats.ncx2.cdf(q, df=df, nc=delta)
        return beta
