from power import (
    PowerBase,
    is_integer
)
import scipy.stats as stats


class Distribution(PowerBase):
    """ """

    # Parameters controling the simulation of power
    _N_SIMS = 1000
    _SEED = 710321

    # Parameters controling the search grid for the calculation of sample size.
    _minN = 2
    _maxN = 1000

    def __init__(self, n=None, alpha=None, beta=None, power=None, method=None,
                 dist=None, compare_dist=None, seed=None, **kwargs):
        if n is not None:
            is_integer(n, '`n` should be of type Int.')
        self.n = n

        if dist in dir(stats):
            self.dist = getattr(stats, dist)(**kwargs)
        else:
            raise ValueError('{} is not a valid distribution'.format(dist))

        if compare_dist not in dir(stats):
            raise ValueError('{} is not a valid distribution'.format(compare_dist))

        if seed is None:
            self.seed = self._SEED
        else:
            self.seed = seed

        if method == 'anderson' or method == 'anderson-darling':
            if compare_dist not in ["norm", "expon", "logistic", "gumbel",
                                    "gumbel_l", "gumbel_r", "extreme1"]:
                raise ValueError('{} is not a valid distribution'.format(compare_dist))

            def dist_anderson(rvs):
                return stats.anderson(rvs, dist=compare_dist)

            self.distribution_test = dist_anderson
        elif method == 'kolmogorov-smirnov' or method == 'ks':
            def dist_ks(rvs):
                return stats.kstest(rvs, dist=compare_dist)
            self.distribution_test = dist_ks

        # Initialize the remaining arguments through the parent.
        super(Distribution, self).__init__(alpha=alpha, power=power,
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

    def calculate_n(self):
        """ Perfrom the power calculation """
        if self.power is None:
            power = 0.8
        else:
            power = self.power

        if self.alpha is None:
            alpha = 0.05
        else:
            alpha = self.alpha

        found_solution = False
        for n in range(self._minN, self._maxN):
            count = 0
            for sim in range(self._N_SIMS):
                res = self.dist.rvs(size=n, random_state=self.seed * sim)

                _, p_val = self.normal_test(res)
                if p_val < alpha:
                    count += 1
            test_power = count / self._N_SIMS
            if test_power > power:
                found_solution = True
                self.power = test_power
                self.n = n
                break
        if not found_solution:
            raise BaseException("N is greater than maximum N")

    def calculate_power(self):
        """ Perfrom the power calculation """
        if self.alpha is None:
            alpha = 0.05
        else:
            alpha = self.alpha

        count = 0
        for sim in range(self._N_SIMS):
            res = self.dist.rvs(size=self.n, random_state=self.seed * sim)

            _, p_val = self.distribution_test(res)
            if p_val < alpha:
                count += 1
        self.power = count / self._N_SIMS
        self.beta = 1 - self.power

    def calculate_alpha(self):
        """ Perfrom the power calculation """
        if self.power is None:
            power = 0.8
        else:
            power = self.power

        p_vals = []
        for sim in range(self._N_SIMS):
            res = self.dist.rvs(size=self.n, random_state=self.seed * sim)

            _, p_val = self.distribution_test(res)
            p_vals.append(p_val)
        p_vals.sort()
        self.alpha = p_vals[int(self._N_SIMS * power) - 1]
