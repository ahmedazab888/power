from power import (
    PowerBase,
    is_integer
)
import scipy.stats as stats
import math


class Normal(PowerBase):
    """ """

    # Parameters controling the simulation of power
    _N_SIMS = 1000
    _SEED = 710321

    # Parameters controling the search grid for the calculation of sample size.
    _minN = 3
    _maxN = 1000

    def __init__(self, n=None, alpha=None, beta=None, power=None, method=None,
                 dist=None, seed=None, **kwargs):
        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        if dist in dir(stats):
            self.dist = getattr(stats, dist)(**kwargs)

        if seed is None:
            self.seed = self._SEED
        else:
            self.seed = seed

        if method == 'anderson' or method == 'anderson-darling':
            # Need to figure out how to do this right
            raise ValueError('{} is not a valid method'.format(method))
        elif method == 'kolmogorov-smirnov' or method == 'ks':
            def norm_ks(rvs):
                return stats.kstest(rvs, 'norm')
            self.normal_test = norm_ks
        elif method == 'kurt' or method == 'kurtosis':
            self.normal_test = stats.kurtosistest
            self._minN = 20
        elif method == 'martinez-iglewicz':
            raise ValueError('{} is not a valid method'.format(method))
        elif method in ['normaltest', 'omnibus']:
            self.normal_test = stats.normaltest
            self._minN = 20
        elif method == 'range':
            raise ValueError('{} is not a valid method'.format(method))
        elif method in ['shapiro', 'sw', 'shapiro-wilks']:
            self.normal_test = stats.shapiro
        elif method in ['skew', 'skewness']:
            self.normal_test = stats.skewtest
            self._minN = 8
        else:
            raise ValueError('{} is not a valid method'.format(method))

        # Initialize the remaining arguments through the parent.
        super(Normal, self).__init__(alpha=alpha, power=power,
                                     beta=beta, hypothesis=None)

    def calculate(self):
        if self.n is None:
            self._set_default_alpha()
            self._set_default_power()
            self._calculate_n()
        elif self.power is None:
            self._set_default_alpha()
            self._calculate_power()
        elif self.alpha is None:
            self._calculate_alpha()

    def _calculate_n(self):
        """ Perfrom the power calculation """
        if self.power is None:
            power = 0.8
        else:
            power = self.power

        if self.alpha is None:
            alpha = 0.5
        else:
            alpha = self.alpha

        found_solution = False
        # Find power at endpoints
        lag_lower = (self._minN, self._power_internals(self._minN, alpha))
        if lag_lower[1] > power:
            self.power = lag_lower[1]
            self.n = lag_lower[0]
            return
        lag_upper = (self._maxN, self._power_internals(self._maxN, alpha))

        if lag_lower[1] > lag_upper[1]:
            # I
            tmp = lag_upper
            lag_upper = lag_lower
            lag_lower = tmp
            if lag_upper[1] < power:
                self.power = lag_upper[1]
                self.n = lag_upper[0]
                return

        if lag_upper[1] < power:
            raise BaseException("N is greater than maximum N")

        while True:
            delta_n = lag_upper[0] - lag_lower[0]
            test_n = math.floor(delta_n / 2) + lag_lower[0]
            test_power = self._power_internals(test_n, alpha)
            if test_power < power:
                # Look at upper half of what's left
                lag_lower = (test_n, test_power)
            else:
                # Otherwise, look at the lower half
                lag_upper = (test_n, test_power)
            if delta_n <= 1:
                # We've zeroed in.  Let's get out of the loop
                found_solution = True
                break

        if not found_solution:
            raise BaseException("N is greater than maximum N")

        if lag_lower[1] > power:
            self.n = lag_lower[0]
            self.power = lag_lower[1]
        else:
            self.n = lag_upper[0]
            self.power = lag_upper[1]
        self.beta = 1 - self.power

    def _calculate_power(self):
        """ Perfrom the power calculation """
        if self.alpha is None:
            alpha = 0.05
        else:
            alpha = self.alpha

        self.power = self._power_internals(self.n, alpha)
        self.beta = 1 - self.power

    def _calculate_alpha(self):
        """ Perfrom the power calculation """
        if self.power is None:
            power = 0.8
        else:
            power = self.power

        p_vals = []
        for sim in range(self._N_SIMS):
            res = self.dist.rvs(size=self.n, random_state=self.seed * sim)

            _, p_val = self.normal_test(res)
            p_vals.append(p_val)
        p_vals.sort()
        self.alpha = p_vals[int(self._N_SIMS * power) - 1]

    def _power_internals(self, n, alpha):
        count = 0
        for sim in range(self._N_SIMS):
            res = self.dist.rvs(size=n, random_state=self.seed * sim)
            _, p_val = self.normal_test(res)
            if p_val < alpha:
                count += 1
        return count / self._N_SIMS
