from power.gof import PearsonIndependance


class LikelihoodRatio(PearsonIndependance):
    """ This is a mask for power.gof.PearsonIndependance.

    The likelihood ratio test for independance is asymptotically equal to
    the Pearson's test of independance.  Therefore, this class masks the
    class utilizing Pearson's test of independance
    """
    pass
