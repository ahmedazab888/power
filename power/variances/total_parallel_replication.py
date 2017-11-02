from power.means import OneSample
import math


class TotalParallelReplication(OneSample):
    """ Tests for Total Variablility using a Parallel study with
    :math:`m` replicates.

    The test for equality tests:
    :math:`H_{0}: \\frac{\\sigma_{TT}}{\\sigma_{TR}} = 1` versus
    :math:`H_{1}: \\frac{\\sigma_{TT}}{\\sigma_{TR}} \\ne 1`.

    The test for superiority tests:
    :math:`H_{0}: \\frac{\\sigma_{TT}}{\\sigma_{TR}} \\ge \\delta` versus
    :math:`H_{1}: \\frac{\\sigma_{TT}}{\\sigma_{TR}} < \\delta`.

    Attributes:
        n: The sample size
        m: The number of replicates
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        stdev_wr: The variance within treatment.
        stdev_br: The variance between replicates.
        stdev_bt: The variance between treatment.
        stdev_wt: The variance within replicates.

    Note:
        This is almost the same as InterSubjectParallel, except the
        adjusted_stdev_tr is total standard deviation, not just between.
    """

    def __init__(self, n=None, m=None, stdev_bt=None, stdev_wt=None,
                 stdev_br=None, stdev_wr=None, similarity_limit=None,
                 hypothesis=None, alpha=None, beta=None, power=None):
        if isinstance(m, int):
            self.m = m
        else:
            raise ValueError('`m` should be of type Int.')

        if hypothesis is 'superiority':
            if similarity_limit is None:
                raise ValueError('You must supply a `similarity_limit` for '
                                 'this hypothesis')
            else:
                sigma_star = 2 * ((stdev_bt**2 + stdev_wt**2 / m)**2 +
                                  similarity_limit**4 *
                                  (stdev_br**2 + stdev_wr**2 / m)**2 +
                                  (stdev_wt**4 / m**2) +
                                  similarity_limit**4 *
                                  (stdev_wr**4 / m**2))
                sigma_star = math.sqrt(sigma_star)
                adjusted_stdev_tr = (math.sqrt(stdev_wr**2 + stdev_br**2) *
                                     similarity_limit)
        elif hypothesis is 'equality':
            sigma_star = 2 * ((stdev_bt**2 + stdev_wt**2 / m)**2 +
                              (stdev_br**2 + stdev_wr**2 / m)**2 +
                              (stdev_wt**4 / m**2) +
                              (stdev_wr**4 / m**2))
            sigma_star = math.sqrt(sigma_star)
            adjusted_stdev_tr = math.sqrt(stdev_wr**2 + stdev_br**2)
        else:
            raise ValueError("`hypothesis` should be a in "
                             "['equality', 'superiority'].")

        stdev_tt = math.sqrt(stdev_bt**2 + stdev_wt**2)
        # Initialize the remaining arguments through the parent.
        super(TotalParallelReplication, self).__init__(n=n,
                                                       mu=stdev_tt**2,
                                                       mu_0=adjusted_stdev_tr**2,
                                                       stdev=sigma_star,
                                                       known_stdev=True,
                                                       alpha=alpha,
                                                       margin=0,
                                                       beta=beta,
                                                       power=power,
                                                       hypothesis=hypothesis)
