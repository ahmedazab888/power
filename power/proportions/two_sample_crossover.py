from power.means import TwoSampleCrossover as TwoSampleCrossoverMeans
from power import is_in_0_1


class TwoSampleCrossover(TwoSampleCrossoverMeans):
    """ Hypotheses for a two sample test of proportions under large sample
        theory and a parallel design.

    The Two Sample Test of Means for Crossover Studies covers three hypothesis:
    equality, superiority, and equivalence.  Let :math:`\\epsilon` be the
    difference between the proportions of interest.

    The test for equality tests :math:`H_{0}: \\epsilon=0` versus
    :math:`H_{1}: \\epsilon \\ne \\mu_{0}`.

    The test for superiority tests :math:`H_{0}: \\epsilon > 0` versus
    :math:`H_{1}: \\epsilon \\le 0`.

    The test for equivalence tests :math:`H_{0}: \\epsilon \\ne 0` versus
    :math:`H_{1}: \\epsilon=0`.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        epsilon: The estimated difference between the probabilities
        stdev: :math:`\\sigma_{d}`, estimated as is as in Chow (section 4.4)
    """

    def __init__(self, n=None, epsilon=None, stdev=None, hypothesis=None,
                 margin=None, alpha=None, beta=None, power=None):

        is_in_0_1(epsilon, '`epsilon` should be in [0, 1]')

        # Initialize the remaining arguments through the parent.
        super(TwoSampleCrossover, self).__init__(n=n, mu_1=epsilon, mu_2=0,
                                                 stdev=stdev, known_stdev=True,
                                                 alpha=alpha, beta=beta,
                                                 power=power, margin=margin,
                                                 hypothesis=hypothesis)
