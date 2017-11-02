from power.gof import GofBase
from power import (is_in_0_1,
                            is_integer)
import math
import scipy.optimize as optimize


class PearsonIndependance(GofBase):
    """ Hypotheses that tests a independance

    Person's Independance Test uses a Chi Squared test to test for the
    independance in a single stratum.

    Attributes:
        n: The sample size
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1 - power`).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        p: A list of lists of observed probabilities.
    """
    def __init__(self, n=None, alpha=None, beta=None, power=None, p=None):
        # This should be made much better (see CMH)
        if isinstance(p, list):
            for values in p:
                if isinstance(values, list):
                    for value in values:
                        is_in_0_1(value,
                                  'All values of p should be in (0, 1).')
                else:
                    raise ValueError("Each list in `p` must be a list "
                                     "of numerics")
        else:
            raise ValueError("`p` must be a list of lists  of numerics")
        self.p = p

        # needed for degree of freedom calculations
        rows = len(p)
        cols = len(p[0])
        self.df = (rows - 1) * (cols - 1)
        self._denom = 0

        # This will be much improved when I implement this as a matrix!
        colsums = [0] * cols
        for i in range(rows):
            for j in range(cols):
                colsums[j] += p[i][j]

        for i in range(rows):
            rowsum = sum(p[i])
            for j in range(cols):
                self._denom += ((p[i][j] - rowsum * colsums[j])**2 /
                                (rowsum * colsums[j]))

        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        # Set remaining variables.
        super(PearsonIndependance, self).__init__(alpha=alpha,
                                                  beta=beta,
                                                  power=power)

    def calculate(self):
        if self.n is None:
            if self.alpha is None:
                self.alpha = 0.05
            if self.power is None:
                power = 0.8
            else:
                power = self.power
            beta = 1 - power

            df = self.df
            delta = optimize.brenth(lambda delta:
                                    self._beta(df, delta, self.alpha) - beta,
                                    a=0.000001, b=1e7)
            n = delta / self._denom
            self.n = math.ceil(n)
            self.beta = beta
            self.power = power
