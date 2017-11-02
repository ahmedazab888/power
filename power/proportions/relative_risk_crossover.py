from power.means import TwoSampleCrossover
from power import is_positive
import math


class RelativeRiskCrossover(TwoSampleCrossover):
    """ Hypotheses for a two sample test of proportions under large sample
        theory and a crossover design using relative risks.

    The Two Sample Test of Proportions for Crossover Studies using Relative
    Risks covers three hypothesis: equality, superiority, and equivalence.
    Let :math:`OR` be the odds ratio equal to:

    :math:`OR=\\frac{p_{2}(1 - p_{1})}{p_{1}(1 - p_{2})}`

    where :math:`p_{2}` is the proportion for the treatment group and
    :math:`p_{1}` is the proportion for the control group.

    The test for equality tests :math:`H_{0}: OR=1` versus
    :math:`H_{1}: OR \\ne 1`.

    The test for superiority tests :math:`H_{0}: OR > 1` versus
    :math:`H_{1}: OR \\le 1`.

    The test for equivalence tests :math:`H_{0}: OR \\ne 1` versus
    :math:`H_{1}: OR = 1`.


    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta` for
            group 1.
        n_1: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta` for
            group 1.
        n_2: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta` for
            group 2.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        hypothesis: One of 'equality', 'superiority', or 'equvalence'.  Tests
            of 'non-inferiority' are the same as tests of 'superiority' with
            respect to the power calculation so choose 'superiority' for both.
        margin: This is the superiority or equivalence margin.
        odds_ratio: The estimated odds ratio
        stdev: :math:`\\sigma_{d}`, estimated as is as in Chow (section 4.7)
    """
    def __init__(self, n=None, odds_ratio=None, stdev=None, hypothesis=None,
                 margin=None, alpha=None, beta=None, power=None):
        is_positive(odds_ratio, '`odds_ratio` should be greater than 0')
        log_odds = math.log(odds_ratio)

        # Initialize the remaining arguments through the parent.
        super(RelativeRiskCrossover, self).__init__(n=n, mu_1=log_odds, mu_2=0,
                                                    stdev=stdev, known_stdev=True,
                                                    alpha=alpha, beta=beta,
                                                    power=power, margin=margin,
                                                    hypothesis=hypothesis)
