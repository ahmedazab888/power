from power.distributions import (Normal,
                                 Distribution)


def test_normal():
    """ From Chow et al. 6.1.2 """

    h = Normal(n=20, power=None, alpha=0.05, method='shapiro', dist='expon', loc=1, scale=1, seed=72)
    h.calculate()
    print(h)
    assert h.n == 23
