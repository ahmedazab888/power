from power.gof import GofBase
from power import (is_in_0_1,
                            is_integer)
import math
import scipy.optimize as optimize


class Pearson(GofBase):
    """ Hypotheses that tests a goodness of fit.

    The Pearon Goodness of Fit test tests the hypothesis that:

    :math:`H_{0}: p_{k} = p_{k,0}` vs.
    :math:`H_{1}: p_{k} \\ne p_{k,0}`

    where :math:`p_{k,0}` is some reference values.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        p_0: A list of expected probabilities
        p: A list of observed probabilities
    """
    def __init__(self, n=None, alpha=None, beta=None, power=None,
                 p_0=None, p=None):
        if isinstance(p, list):
            for value in p:
                is_in_0_1(value, 'All values of p should be in (0, 1).')
        else:
            raise ValueError("`p` must be a list of Numerics.")
        self.p = p

        if isinstance(p_0, list):
            for value in p_0:
                is_in_0_1(value, 'All values of p_0 should be in (0, 1).')
        else:
            raise ValueError("`p_0` must be a list of Numerics.")
        self.p_0 = p_0

        if not len(p) == len(p_0):
            raise ValueError("`p_0` and `p` must have the same length.")

        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        # Set remaining variables.
        super(Pearson, self).__init__(alpha=alpha, beta=beta, power=power)

    def calculate(self):
        p = self.p
        p_0 = self.p_0

        denom = sum([(i - j)**2 / j for i, j in zip(p, p_0)])

        if self.n is None:
            if self.alpha is None:
                self.alpha = 0.05
            if self.power is None:
                power = 0.8
            else:
                power = self.power
            beta = 1 - power
            df = len(p) - 1
            delta = optimize.brenth(lambda delta:
                                    self._beta(df, delta, self.alpha) - beta,
                                    a=1e-7, b=1e4)
            n = delta / denom
            self.n = math.ceil(n)
            self.beta = beta
            self.power = power
