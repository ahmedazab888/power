from power.means import TwoSampleCrossover


class Average(TwoSampleCrossover):
    """ Power and Sample Size calculations for tests of average bioequivalence.

    This class calculates the power and sample size for testing Average
    Bioequivalence using a two-sequence, two-period crossover design.  The
    hypothesis thus is the same as an equivalence test using the
    TwoSampleCrossover class.

    Attributes:
        n: The sample size required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        delta: :math:`\\delta` is the average bioequivalence
        stdev: :math:`\\sigma` is the standard deviation of the sample.
        known_stdev: A boolean indicator if the standard deviation
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
    """

    def __init__(self, n=None, delta=None, stdev=None, known_stdev=None,
                 margin=None, alpha=None, beta=None, power=None):

        # Initialize the remaining arguments through the parent.
        super(Average, self).__init__(n=n, mu_1=delta, mu_2=0,
                                      stdev=stdev, known_stdev=known_stdev,
                                      hypothesis="equivalence",
                                      margin=margin, alpha=alpha,
                                      beta=beta, power=power)
