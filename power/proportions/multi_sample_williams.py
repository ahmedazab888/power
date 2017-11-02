from power.means import MultiSampleWilliams as \
    MultiSampleWilliamsMeans
from power import is_in_0_1


class MultiSampleWilliams(MultiSampleWilliamsMeans):
    """ Power and Sample Size calculations for multi-sample Williams Design
    for proportions.

    The test for equality tests :math:`H_{0}: \\epsilon = 0` versus
    :math:`H_{1}: \\epsilon \\ne 0`.

    The test for superiority tests :math:`H_{0}: \\epsilon > 0` versus
    :math:`H_{1}: \\epsilon \\le 0`.

    The test for equivalence tests :math:`H_{0}: \\epsilon \\ne 0` versus
    :math:`H_{1}: \\epsilon = 0`.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        p: :math:`p` is a list of proportions to be tested.
        sigma: :math:`\\sigma` is the standard deviation of the sample.
        known_stdev: A boolean indicator if the standard deviation
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        hypothesis: One of 'equality', 'superiority', or 'equvalence'.  Tests
            of 'non-inferiority' are the same as tests of 'superiority' with
            respect to the power calculation so choose 'superiority' for both.
        margin: This is the superiority or equivalence margin.

    Note:
        Currently, this function returns the best power and the best alpha when
        power or alpha are unknown.  I don't think this is the best approach,
        so it may change.  I may change it to return a power calculation for
        each contrast.
    """

    def __init__(self, n=None, p=None, stdev=None, hypothesis=None,
                 margin=None, alpha=None, beta=None, power=None):

        for value in p:
            is_in_0_1(value, 'Each value in `p` should be in [0, 1].')

        # This is the same as the Multi-Sample Williams design for means
        super(MultiSampleWilliams, self).__init__(n=n,
                                                  mu=p,
                                                  stdev=stdev,
                                                  margin=margin,
                                                  alpha=alpha,
                                                  power=power,
                                                  beta=beta,
                                                  known_stdev=True,
                                                  hypothesis=hypothesis)
