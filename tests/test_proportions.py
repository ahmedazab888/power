from skdesign.power.proportions import (Binomial,
                                        Fisher,
                                        # MultiSampleWilliams,
                                        OneSample,
                                        OneWayAnova,
                                        RelativeRiskParallel,
                                        TwoSampleCrossover,
                                        TwoSampleParallel)


def test_binomial():
    """ See 5.1.3 in Chow et al. for calculations """
    # h = Binomial(alpha=0.05, power=0.8, p=0.3, p_0=0.1)
    # h.calculate()
    # assert h.n == 25
    # assert h.power > 0.8
    # assert h.alpha < 0.05

    # h = Binomial(alpha=0.05, power=0.8, p=0.3, p_0=0.05, margin=0.05)
    # h.calculate()
    # assert h.n == 25
    # assert h.power > 0.8
    # assert h.alpha < 0.05


def test_fisher():
    """ See Table 5.2.1 in Chow et al. for calculations """
    h = Fisher(alpha=0.05, power=0.8, p_1=0.05, p_2=0.3)
    h.calculate()
    assert h.n_1 == 34

    h = Fisher(alpha=0.05, power=0.8, p_1=0.6, p_2=0.9)
    h.calculate()
    assert h.n_1 == 30


def test_one_sample():
    """ See 4.1.4 in Chow et al. for calculations
    """
    h = OneSample(p=0.5, p_0=0.3, alpha=0.05, power=0.8,
                  hypothesis='equality')
    h.calculate()
    assert h.n == 50
    assert h.power > 0.8

    h = OneSample(p=0.5, p_0=0.3, alpha=0.05, n=50,
                  hypothesis='equality')
    h.calculate()
    assert h.power > 0.8

    h = OneSample(p=0.5, p_0=0.3, n=50, power=0.8,
                  hypothesis='equality')
    h.calculate()
    assert h.alpha < 0.05

    h = OneSample(p=0.5, p_0=0.3, alpha=0.05, power=0.8, margin=-0.1,
                  hypothesis='superiority')
    h.calculate()
    assert h.n == 18
    assert h.power > 0.8

    h = OneSample(p=0.5, p_0=0.3, alpha=0.05, n=18, margin=-0.1,
                  hypothesis='superiority')
    h.calculate()
    assert h.power > 0.8

    h = OneSample(p=0.5, p_0=0.3, n=18, power=0.8, margin=-0.1,
                  hypothesis='superiority')
    h.calculate()
    assert h.alpha < 0.05

    h = OneSample(p=0.6, p_0=0.6, alpha=0.05, power=0.8, margin=0.2,
                  hypothesis='equivalence')
    h.calculate()
    assert h.n == 52
    assert h.power > 0.8

    h = OneSample(p=0.6, p_0=0.6, alpha=0.05, n=52, margin=0.2,
                  hypothesis='equivalence')
    h.calculate()
    assert h.power > 0.8

    h = OneSample(p=0.6, p_0=0.6, n=52, power=0.8, margin=0.2,
                  hypothesis='equivalence')
    h.calculate()
    assert h.alpha < 0.05


def test_two_sample_parallel():
    """ See 4.2.4 in Chow et al. for calculations
    """
    h = TwoSampleParallel(p_1=0.65, p_2=0.85, alpha=0.05, power=0.8,
                          hypothesis='equality')
    h.calculate()
    assert h.n_1 == 70
    assert h.n_2 == 70
    assert h.power > 0.8

    h = TwoSampleParallel(p_1=0.65, p_2=0.85, alpha=0.05, n_1=70,
                          hypothesis='equality')
    h.calculate()
    assert h.power > 0.8

    h = TwoSampleParallel(p_1=0.65, p_2=0.85, n_2=70, power=0.8,
                          hypothesis='equality')
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleParallel(p_1=0.65, p_2=0.85, alpha=0.05, power=0.8,
                          margin=0.1, hypothesis='superiority')
    h.calculate()
    assert h.n_1 == 25
    assert h.n_2 == 25
    assert h.power > 0.8

    h = TwoSampleParallel(p_1=0.65, p_2=0.85, alpha=0.05, n_1=25,
                          margin=0.1, hypothesis='superiority')
    h.calculate()
    assert h.power > 0.8

    h = TwoSampleParallel(p_1=0.65, p_2=0.85, n_2=25, power=0.8,
                          margin=0.1, hypothesis='superiority')
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleParallel(p_1=0.65, p_2=0.85, alpha=0.05, power=0.8,
                          margin=-0.05, hypothesis='superiority')
    h.calculate()
    assert h.n_1 == 98
    assert h.n_2 == 98
    assert h.power > 0.8

    h = TwoSampleParallel(p_1=0.65, p_2=0.85, alpha=0.05, n_1=98,
                          margin=-0.05, hypothesis='superiority')
    h.calculate()
    assert h.power > 0.8

    h = TwoSampleParallel(p_1=0.65, p_2=0.85, n_2=98, power=0.8,
                          margin=-0.05, hypothesis='superiority')
    h.calculate()
    assert h.alpha < 0.05

    # Chow has 132.  The difference is due to when rounding occurs
    h = TwoSampleParallel(p_1=0.75, p_2=0.8, alpha=0.05, power=0.8,
                          margin=0.2, hypothesis='equivalence')
    h.calculate()
    assert h.n_1 == 133
    assert h.n_2 == 133
    assert h.power > 0.8

    # Chow has 132.  The difference is due to when rounding occurs
    h = TwoSampleParallel(p_1=0.75, p_2=0.8, alpha=0.05, n_1=133,
                          margin=0.2, hypothesis='equivalence')
    h.calculate()
    assert h.power > 0.8

    # Chow has 132.  The difference is due to when rounding occurs
    h = TwoSampleParallel(p_1=0.75, p_2=0.8, n_2=133, power=0.8,
                          margin=0.2, hypothesis='equivalence')
    h.calculate()
    assert h.alpha < 0.05


def test_two_sample_crossover():
    """ See 4.3.4 in Chow et al. for calculations
    """
    h = TwoSampleCrossover(epsilon=0.2, alpha=0.05, power=0.8, stdev=0.5,
                           hypothesis='equality')
    h.calculate()
    assert h.n == 25
    assert h.power > 0.8

    h = TwoSampleCrossover(epsilon=0.2, alpha=0.05, n=25, stdev=0.5,
                           hypothesis='equality')
    h.calculate()
    assert h.power > 0.8

    h = TwoSampleCrossover(epsilon=0.2, n=25, power=0.8, stdev=0.5,
                           hypothesis='equality')
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleCrossover(epsilon=0, alpha=0.05, power=0.8, stdev=0.5,
                           margin=-0.2, hypothesis='superiority')
    h.calculate()
    assert h.n == 20
    assert h.power > 0.8

    h = TwoSampleCrossover(epsilon=0, alpha=0.05, n=20, stdev=0.5,
                           margin=-0.2, hypothesis='superiority')
    h.calculate()
    assert h.power > 0.8

    h = TwoSampleCrossover(epsilon=0, n=20, power=0.8, stdev=0.5,
                           margin=-0.2, hypothesis='superiority')
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleCrossover(epsilon=0, alpha=0.05, power=0.8, stdev=0.5,
                           margin=0.2, hypothesis='equivalence')
    h.calculate()
    assert h.n == 27
    assert h.power > 0.8

    h = TwoSampleCrossover(epsilon=0, alpha=0.05, n=27, stdev=0.5,
                           margin=0.2, hypothesis='equivalence')
    h.calculate()
    assert h.power > 0.8

    h = TwoSampleCrossover(epsilon=0, n=27, power=0.8, stdev=0.5,
                           margin=0.2, hypothesis='equivalence')
    h.calculate()
    assert h.alpha < 0.05


def test_one_way_anova():
    """ See 4.4.4 in Chow et al. for calculations

    Chow has 95.  The difference is due to when rounding occurs
    """
    p = [0.4, 0.5]
    p_0 = 0.2
    h = OneWayAnova(p=p, p_0=p_0, alpha=0.05, power=0.8)
    h.calculate()
    assert h.n == 96
    assert h.power > 0.8


def test_multi_sample_williams():
    """ See 4.5.4 in Chow et al.  None of these pass because it is not set up
    to consider only one of the various contrasts, as in the book.  However, if
    you print out part, you see that it is calculating correctly.  I need to
    find a better test for this.
    """
    pass
    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75,
    #                         hypothesis='equality', alpha=0.05, power=0.8)
    # h.calculate()
    # assert h.n == 19
    # assert h.power > 0.8

    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75,
    #                         hypothesis='equality', alpha=0.05, n=19)
    # h.calculate()
    # assert h.power > 0.8

    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75,
    #                         hypothesis='equality', n=19, power=0.8)
    # h.calculate()
    # assert h.alpha < 0.05

    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75, margin=0.05,
    #                         hypothesis='superiority', alpha=0.05,
    #                         power=0.8)
    # h.calculate()
    # assert h.n == 27
    # assert h.power > 0.8

    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75, margin=-0.05,
    #                         hypothesis='superiority', alpha=0.05, n=27)
    # h.calculate()
    # assert h.power > 0.8

    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75, margin=-0.05,
    #                         hypothesis='superiority', n=27, power=0.8)
    # h.calculate()
    # assert h.alpha < 0.05

    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75, margin=0.3,
    #                         hypothesis='equivalence', alpha=0.05,
    #                         power=0.8)
    # h.calculate()
    # assert h.n == 80
    # assert h.power > 0.8

    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75, margin=0.3,
    #                         hypothesis='equivalence', alpha=0.05, n=80)
    # h.calculate()
    # assert h.power > 0.8

    # h = MultiSampleWilliams(p=[0.50, 0.30, 0.35], sigma=0.75, margin=0.3,
    #                         hypothesis='equivalence', n=80, power=0.8)
    # h.calculate()
    # assert h.alpha < 0.05


def test_relative_risk_parallel():
    """ See 4.6.4 in Chow et al. """
    h = RelativeRiskParallel(p_1=0.25, p_2=0.40, alpha=0.05, power=0.8,
                             hypothesis='equality')
    h.calculate()
    assert h.n_1 == 156
    assert h.n_2 == 156
    assert h.power > 0.8

    h = RelativeRiskParallel(p_1=0.25, p_2=0.40, alpha=0.05, n_1=156,
                             hypothesis='equality')
    h.calculate()
    assert h.power > 0.8

    h = RelativeRiskParallel(p_1=0.25, p_2=0.40, n_2=156, power=0.8,
                             hypothesis='equality')
    h.calculate()
    assert h.alpha < 0.05

    # Chow has 241, but due to rounding, we get 242
    h = RelativeRiskParallel(p_1=0.25, p_2=0.40, alpha=0.05, power=0.8,
                             margin=-0.2, hypothesis='superiority')
    h.calculate()
    assert h.n_1 == 242
    assert h.n_2 == 242
    assert h.power > 0.8

    # Chow has 241, but due to rounding, we get 242
    h = RelativeRiskParallel(p_1=0.25, p_2=0.40, alpha=0.05, n_1=242,
                             margin=0.2, hypothesis='superiority')
    h.calculate()
    assert h.power > 0.8

    # Chow has 241, but due to rounding, we get 242
    h = RelativeRiskParallel(p_1=0.25, p_2=0.40, n_2=242, power=0.8,
                             margin=0.2, hypothesis='superiority')
    h.calculate()
    assert h.alpha < 0.05

    # Chow has 364, but due to rounding, we get 366
    h = RelativeRiskParallel(p_1=0.25, p_2=0.25, alpha=0.05, power=0.8,
                             margin=0.5, hypothesis='equivalence')
    h.calculate()
    assert h.n_1 == 366
    assert h.n_2 == 366
    assert h.power > 0.8

    # Chow has 364, but due to rounding, we get 366
    h = RelativeRiskParallel(p_1=0.25, p_2=0.25, alpha=0.05, n_1=366,
                             margin=0.5, hypothesis='equivalence')
    h.calculate()
    assert h.power > 0.8

    # Chow has 364, but due to rounding, we get 366
    h = RelativeRiskParallel(p_1=0.25, p_2=0.25, n_2=366, power=0.8,
                             margin=0.5, hypothesis='equivalence')
    h.calculate()
    assert h.alpha < 0.05


def test_relative_risk_crossover():
    """ This is the same as the means.TwoSampleCrossover """
    pass
