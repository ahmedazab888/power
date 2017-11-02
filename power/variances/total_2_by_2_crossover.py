from power.means import OneSample
import math


class Total2By2Crossover(OneSample):
    """ Tests for Intersubject Variablility using a 2 by 2 Crossover study

    The test for equality tests:
    :math:`H_{0}: \\frac{\\sigma_{TT}}{\\sigma_{TR}} = 1` versus
    :math:`H_{1}: \\frac{\\sigma_{TT}}{\\sigma_{TR}} \\ne 1`.

    The test for superiority tests:
    :math:`H_{0}: \\frac{\\sigma_{TT}}{\\sigma_{TR}} \\ge \\delta` versus
    :math:`H_{1}: \\frac{\\sigma_{TT}}{\\sigma_{TR}} < \\delta`.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        stdev_wr: The total variance.
        stdev_br: The variance between replicates.
        stdev_bt: The variance between treatment.
        stdev_wt: The total variance.
        rho: correlation
    """

    def __init__(self, n=None, stdev_bt=None, stdev_wt=None, rho=None,
                 stdev_br=None, stdev_wr=None, similarity_limit=None,
                 hypothesis=None, alpha=None, beta=None, power=None):

        stdev_tt = math.sqrt(stdev_bt**2 + stdev_wt**2)
        stdev_tr = math.sqrt(stdev_br**2 + stdev_wr**2)
        if hypothesis is 'superiority':
            if similarity_limit is None:
                raise ValueError('You must supply a `similarity_limit` for '
                                 'this hypothesis')
            else:
                sigma_star = 2 * (stdev_tt**4 +
                                  similarity_limit**4 * stdev_tr**4 -
                                  2 * rho**2 * stdev_bt**2 * stdev_br**2 *
                                  similarity_limit**2)
                sigma_star = math.sqrt(sigma_star)
                adjusted_stdev_tr = stdev_tr * similarity_limit
        elif hypothesis is 'equality':
            sigma_star = 2 * (stdev_tt**4 + stdev_tr**4 -
                              2 * rho**2 * stdev_bt**2 * stdev_br**2)
            sigma_star = math.sqrt(sigma_star)
            adjusted_stdev_tr = stdev_tr
        else:
            raise ValueError("`hypothesis` should be a in "
                             "['equality', 'superiority'].")

        # Initialize the remaining arguments through the parent.
        super(Total2By2Crossover, self).__init__(n=n,
                                                 mu=stdev_tt**2,
                                                 mu_0=adjusted_stdev_tr**2,
                                                 stdev=sigma_star,
                                                 known_stdev=True,
                                                 alpha=alpha,
                                                 margin=0,
                                                 beta=beta,
                                                 power=power,
                                                 hypothesis=hypothesis)
