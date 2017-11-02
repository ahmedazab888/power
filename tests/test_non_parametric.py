from skdesign.power.non_parametric import (OneSample,
                                           TwoSample,
                                           Independance)


def test_one_sample():
    """ See Chow et al. 14.2 """
    h = OneSample(n=None, alpha=0.05, power=0.8,
                  p_2=0.30, p_3=0.40, p_4=0.05)
    h.calculate()
    assert h.n == 383
    assert h.power > 0.8

    h = OneSample(n=383, alpha=0.05, power=None,
                  p_2=0.30, p_3=0.40, p_4=0.05)
    h.calculate()
    assert h.power > 0.8

    h = OneSample(n=383, alpha=None, power=0.8,
                  p_2=0.30, p_3=0.40, p_4=0.05)
    h.calculate()
    assert h.alpha < 0.05


def test_two_sample():
    """ See 14.3 Example from Chow et al. """

    h = TwoSample(n_2=None, alpha=0.05, power=0.8,
                  p_1=0.7, p_2=0.8, p_3=0.8)
    h.calculate()
    assert h.n_1 == 54
    assert h.n_2 == 54
    assert h.power > 0.8

    h = TwoSample(n_1=54, alpha=0.05, power=None,
                  p_1=0.7, p_2=0.8, p_3=0.8)
    h.calculate()
    assert h.power > 0.8

    h = TwoSample(n_2=54, alpha=None, power=0.8,
                  p_1=0.7, p_2=0.8, p_3=0.8)
    h.calculate()
    assert h.alpha < 0.05


def test_independance():
    """ See 14.4 Example from Chow et al. """

    h = Independance(n=None, alpha=0.05, power=0.8,
                     p_1=0.6, p_2=0.7)
    h.calculate()
    assert h.n == 135
    assert h.power > 0.8

    h = Independance(n=135, alpha=0.05, power=None,
                     p_1=0.6, p_2=0.7)
    h.calculate()
    assert h.power > 0.8

    h = Independance(n=135, alpha=None, power=0.8,
                     p_1=0.6, p_2=0.7)
    h.calculate()
    assert h.alpha < 0.05
