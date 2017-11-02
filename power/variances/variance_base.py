from power import (PowerBase,
                            is_positive)


class VarianceBase(PowerBase):
    """ The base for hypothesis on variance using an F statistic.

    Attributes:
        n: The number of subjects required to test the hypothesis at an
            :math:`\\alpha` level and a power of :math:`1 - \\beta`.
        m: The number of replicates per subject.
        stdev_1: The variance of the first treament.
        stdev_2: The variance of the second treamment.
        similarity_limit: :math:`\\delta`

        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
    """

    def __init__(self, n=None, m=None, stdev_1=None, stdev_2=None,
                 similarity_limit=None, hypothesis=None,
                 alpha=None, beta=None, power=None):

        if n is not None:
            if isinstance(n, int):
                self.n = n
            else:
                raise ValueError('`n` should be of type Int.')
        else:
            self.n = None

        if m is not None:
            if isinstance(m, int):
                self.m = m
            else:
                raise ValueError('`m` should be of type Int.')
        else:
            raise ValueError('`m` must be provided.')

        is_positive(stdev_1, 'stdev_1')
        self.stdev_1 = stdev_1
        is_positive(stdev_2, 'stdev_2')
        self.stdev_2 = stdev_2

        # _alpha and _beta are adjusted to be used as cutpoints.
        # alpha and beta are to be used for display
        # We use _beta to prevent floating point errors when beta is passed on
        if hypothesis == 'equality':
            if alpha is not None:
                self._alpha = alpha / 2
            else:
                self._alpha = None
            if power is not None:
                self._beta = 1 - power
            elif beta is not None:
                self._beta = beta
            else:
                self._beta = None
            similarity_limit = 1
            sigma_ratio = stdev_2 / stdev_1
        elif hypothesis == 'superiority':
            if alpha is not None:
                self._alpha = alpha
            else:
                self._alpha = None
            if power is not None:
                self._beta = 1 - power
            elif beta is not None:
                self._beta = beta
            else:
                self._beta = None
            if similarity_limit is None:
                raise ValueError('A similarity_limit must be provided')
            sigma_ratio = stdev_2 / (stdev_1 * similarity_limit)
        elif hypothesis == 'equivalence':
            if alpha is not None:
                self._alpha = 1 - alpha
            else:
                self._alpha = None
            if power is not None:
                self._beta = (power) / 2
            elif beta is not None:
                self._beta = (1 - beta) / 2
            else:
                self._beta = None
            if similarity_limit is None:
                raise ValueError('A similarity_limit must be provided')
            sigma_ratio = stdev_2 / (stdev_1 * similarity_limit)

        self.similarity_limit = similarity_limit
        self.sigma_ratio = sigma_ratio

        # Initialize the remaining arguments through the parent.
        super(VarianceBase, self).__init__(alpha=alpha, power=power,
                                           beta=beta, hypothesis=hypothesis)

    def update_alpha(self):
        """ Transform _alpha to alpha """
        if self.hypothesis == 'equality':
            self.alpha = self._alpha * 2
        elif self.hypothesis == 'superiority':
            self.alpha = self._alpha
        elif self.hypothesis == 'equivalence':
            self.alpha = 1 - self._alpha

    def update_beta(self):
        """ Transform _beta to alpha """
        if self.hypothesis == 'equality':
            self.beta = self._beta
            self.power = 1 - self.beta
        elif self.hypothesis == 'superiority':
            self.beta = self._beta
            self.power = 1 - self.beta
        elif self.hypothesis == 'equivalence':
            self.beta = 1 - 2 * self._beta
            self.power = 1 - self.beta
