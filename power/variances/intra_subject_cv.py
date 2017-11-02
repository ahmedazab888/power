from power.means import OneSample
import math


class IntraSubjectCV(OneSample):
    """ Hypotheses for a testing the equality of Intra-Subject CVs

    There two types of models that can be used: a simple random effects model
    and a conditional random effects model.  The test is the same as a one
    sample test of means with :math:`\\mu=CV_{T}` and :math:`\\mu_{0}=CV_{R}`
    where :math:`CV_{T}` and :math:`CV_{R}` are the CVs for the two treatments.

    For a conditional random effect model, the standard deviations of the
    treatments must be stated.  For a simple random effects model, they are
    calculated from the CVs.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        stdev_t: The variance of the first treament.
        stdev_r: The variance of the second treamment.
    """

    def __init__(self, n=None, m=None, cv_t=None, cv_r=None, stdev_t=None,
                 stdev_r=None, alpha=None, beta=None, power=None,
                 hypothesis=None, margin=None, model=None):

        if model is None:
            model = 'simple'
        elif model not in ['simple', 'conditional']:
            raise ValueError("`model` must be either 'simple' "
                             "or 'conditional'")

        if model == 'simple':
            stdev = cv_t**2 / m**2 + cv_t**4
            stdev += cv_r**2 / m**2 + cv_r**4
            stdev = math.sqrt(stdev)
        else:
            stdev = math.sqrt(stdev_r**2 + stdev_t**2)

        # Initialize the remaining arguments through the parent.
        super(IntraSubjectCV, self).__init__(n=n,
                                             mu=cv_t,
                                             mu_0=cv_r,
                                             stdev=stdev,
                                             known_stdev=True,
                                             alpha=alpha,
                                             beta=beta,
                                             power=power,
                                             margin=margin,
                                             hypothesis=hypothesis)
