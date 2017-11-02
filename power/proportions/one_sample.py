from power.means import OneSample as OneSampleMeans
from power import is_in_0_1
import math


class OneSample(OneSampleMeans):
    """ Hypotheses for a one sample test of proportions under large sample
        theory.

    The One Sample Test of Proportions covers three hypothesis: equality,
    superiority, and equivalence.  Let :math:`p` be the unknown proportion and
    :math:`p_{0}` be the proportion that it is being tested against.

    The test for equality tests :math:`H_{0}: p = p_{0}` versus
    :math:`H_{1}: p \\ne p_{0}`.

    The test for superiority tests :math:`H_{0}: p > p_{0}` versus
    :math:`H_{1}: p \\le p_{0}`.

    The test for equivalence tests :math:`H_{0}: p \\ne p_{0}` versus
    :math:`H_{1}: p = p_{0}`.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        p_0: The null value for the probability
        p: The estimated value for the probability
        hypothesis: One of 'equality', 'superiority', or 'equvalence'.  Tests
            of 'non-inferiority' are the same as tests of 'superiority' with
            respect to the power calculation so choose 'superiority' for both.
        margin: This is the superiority or equivalence margin.
    """

    def __init__(self, n=None, p=None, p_0=None, margin=None,
                 alpha=None, beta=None, power=None, hypothesis=None):

        is_in_0_1(p, 'p')
        self.p = p

        is_in_0_1(p_0, 'p_0')
        self.p_0 = p_0

        stdev = math.sqrt(p * (1 - p))

        # Initialize the remaining arguments through the parent.
        super(OneSample, self).__init__(n=n, mu=p, mu_0=p_0, stdev=stdev,
                                        known_stdev=True, alpha=alpha,
                                        beta=beta, power=power, margin=margin,
                                        hypothesis=hypothesis)
