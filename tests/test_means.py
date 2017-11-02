""" Test cases for the power.means module """

# import pytest
from power.means import (OneSample,
                         TwoSampleParallel,
                         TwoSampleCrossover,
                         OneWayAnova,
                         MultiSampleWilliams)


def test_one_sample():
    """ Test Cases for One Sample test of means """

    # From Chow et al. 3.1.4
    h = OneSample(mu=1, mu_0=0.5, stdev=1, alpha=0.05,
                  power=0.8, hypothesis="equality")
    h.calculate()
    assert h.n == 32
    assert h.power > 0.8

    h = OneSample(mu=1, mu_0=0.5, stdev=1, n=32,
                  power=0.8, hypothesis="equality")
    h.calculate()
    assert h.alpha < 0.05

    h = OneSample(mu=1, mu_0=0.5, stdev=1, n=32,
                  alpha=0.05, hypothesis="equality")
    h.calculate()
    assert h.power > 0.8

    h = OneSample(mu=1, mu_0=0.5, stdev=1, alpha=0.05,
                  power=0.8, margin=-0.5, hypothesis="superiority")
    h.calculate()
    assert h.n == 7
    assert h.power > 0.8

    h = OneSample(mu=1, mu_0=0.5, stdev=1, n=7,
                  power=0.8, margin=-0.5, hypothesis="superiority")
    h.calculate()
    assert h.alpha < 0.05

    h = OneSample(mu=1, mu_0=0.5, stdev=1, n=7,
                  alpha=0.05, margin=-0.5, hypothesis="superiority")
    h.calculate()
    assert h.power > 0.8

    h = OneSample(mu=1, mu_0=0.5, stdev=1, alpha=0.05,
                  power=0.8, margin=-0.5, hypothesis="superiority",
                  known_stdev=False)
    h.calculate()
    assert h.n == 8
    assert h.power > 0.8

    h = OneSample(mu=1, mu_0=0.5, stdev=1, n=8,
                  power=0.8, margin=-0.5, hypothesis="superiority",
                  known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05

    h = OneSample(mu=1, mu_0=1, stdev=0.1, alpha=0.05,
                  power=0.8, margin=0.05, hypothesis="equivalence")
    h.calculate()
    assert h.n == 35
    assert h.power > 0.8

    h = OneSample(mu=1, mu_0=1, stdev=0.1, n=35,
                  power=0.8, margin=0.05, hypothesis="equivalence")
    h.calculate()
    assert h.alpha < 0.05

    h = OneSample(mu=1, mu_0=1, stdev=0.1, n=35,
                  alpha=0.05, margin=0.05, hypothesis="equivalence")
    h.calculate()
    assert h.power > 0.8

    h = OneSample(mu=1, mu_0=1, stdev=0.1, alpha=0.05,
                  power=0.9, margin=0.05, hypothesis="equivalence",
                  known_stdev=False)
    h.calculate()
    assert h.n == 36
    assert h.power > 0.8

    h = OneSample(mu=1, mu_0=1, stdev=0.1, n=36,
                  power=0.9, margin=0.05, hypothesis="equivalence",
                  known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05

    h = OneSample(mu=1, mu_0=1, stdev=0.1, alpha=0.05,
                  n=36, margin=0.05, hypothesis="equivalence",
                  known_stdev=False)
    h.calculate()
    assert h.power > 0.9

    # From PASS 11 Manual pg 400-11
    h = OneSample(mu=2475, mu_0=3300, stdev=663, alpha=0.05,
                  power=0.8, hypothesis="equality",
                  known_stdev=False)
    h.calculate()
    assert h.n == 8
    assert h.power > 0.8

    h = OneSample(mu=2475, mu_0=3300, stdev=663, alpha=0.05,
                  power=0.9, hypothesis="equality",
                  known_stdev=False)
    h.calculate()
    assert h.n == 9
    assert h.power > 0.9

    h = OneSample(mu=2970, mu_0=3300, stdev=663, alpha=0.05,
                  power=0.8, hypothesis="equality",
                  known_stdev=False)
    h.calculate()
    assert h.n == 34
    assert h.power > 0.8

    h = OneSample(mu=2970, mu_0=3300, stdev=663, alpha=0.05,
                  power=0.9, hypothesis="equality",
                  known_stdev=False)
    h.calculate()
    assert h.n == 45
    assert h.power > 0.9

    h = OneSample(mu=3135, mu_0=3300, stdev=663, alpha=0.05,
                  power=0.8, hypothesis="equality",
                  known_stdev=False)
    h.calculate()
    assert h.n == 129
    assert h.power > 0.8

    h = OneSample(mu=3135, mu_0=3300, stdev=663, alpha=0.05,
                  power=0.9, hypothesis="equality",
                  known_stdev=False)
    h.calculate()
    assert h.n == 172
    assert h.power > 0.9

    h = OneSample(mu=2475, mu_0=3300, stdev=663, alpha=0.05,
                  n=8, hypothesis="equality",
                  known_stdev=False)
    h.calculate()
    assert h.power > 0.8

    h = OneSample(mu=2475, mu_0=3300, stdev=663, power=0.8,
                  n=8, hypothesis="equality",
                  known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05


def test_two_sample_parallel():
    """ Tests for the TwoSampleParallel class """

    # From Chow et al. 3.2.5
    h = TwoSampleParallel(mu_1=0.05, mu_2=0, stdev=0.10, alpha=0.05,
                          power=0.8, hypothesis="equality",
                          known_stdev=True)
    h.calculate()
    assert h.n_1 == 63
    assert h.n_2 == 63
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0.05, mu_2=0, stdev=0.10, alpha=0.05,
                          n_2=63, hypothesis="equality",
                          known_stdev=True)
    h.calculate()
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0.05, mu_2=0, stdev=0.10, n_1=63,
                          power=0.8, hypothesis="equality",
                          known_stdev=True)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleParallel(mu_1=0.05, mu_2=0, stdev=0.10, alpha=0.05,
                          power=0.8, hypothesis="equality",
                          known_stdev=False)
    h.calculate()
    assert h.n_1 == 64
    assert h.n_2 == 64
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0.05, mu_2=0, stdev=0.10, alpha=0.05,
                          n_2=64, hypothesis="equality",
                          known_stdev=False)
    h.calculate()
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0.05, mu_2=0, stdev=0.10, n_2=64,
                          power=0.8, hypothesis="equality",
                          known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleParallel(mu_1=0, mu_2=0, stdev=0.10, alpha=0.05,
                          power=0.8, hypothesis="superiority",
                          margin=0.05, known_stdev=True)
    h.calculate()
    assert h.n_1 == 50
    assert h.n_2 == 50
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0, mu_2=0, stdev=0.10, alpha=0.05,
                          n_1=50, n_2=50, hypothesis="superiority",
                          margin=0.05, known_stdev=True)
    h.calculate()
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0, mu_2=0, stdev=0.10, n_2=50, ratio=1,
                          power=0.8, hypothesis="superiority",
                          margin=0.05, known_stdev=True)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleParallel(mu_1=0, mu_2=0, stdev=0.10, alpha=0.05,
                          power=0.8, hypothesis="superiority",
                          margin=0.05, known_stdev=False)
    h.calculate()
    assert h.n_1 == 51
    assert h.n_2 == 51
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0, mu_2=0, stdev=0.10, alpha=0.05,
                          n_1=51, ratio=1, hypothesis="superiority",
                          margin=0.05, known_stdev=False)
    h.calculate()
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0, mu_2=0, stdev=0.10, n_1=51,
                          power=0.8, hypothesis="superiority",
                          margin=0.05, known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleParallel(mu_1=0.01, mu_2=0, stdev=0.10, alpha=0.05,
                          power=0.8, hypothesis="equivalence",
                          margin=0.05, known_stdev=True)
    h.calculate()
    assert h.n_1 == 108
    assert h.n_2 == 108
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0.01, mu_2=0, stdev=0.10, alpha=0.05,
                          n_1=108, hypothesis="equivalence",
                          margin=0.05, known_stdev=True)
    h.calculate()
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0.01, mu_2=0, stdev=0.10, n_1=108,
                          power=0.8, hypothesis="equivalence",
                          margin=0.05, known_stdev=True)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleParallel(mu_1=0.01, mu_2=0, stdev=0.10, alpha=0.05,
                          power=0.8, hypothesis="equivalence",
                          margin=0.05, known_stdev=False)
    h.calculate()
    assert h.n_1 == 108
    assert h.n_2 == 108
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0.01, mu_2=0, stdev=0.10, alpha=0.05,
                          n_2=108, hypothesis="equivalence",
                          margin=0.05, known_stdev=False)
    h.calculate()
    assert h.power > 0.80

    h = TwoSampleParallel(mu_1=0.01, mu_2=0, stdev=0.10, n_2=108,
                          power=0.8, hypothesis="equivalence",
                          margin=0.05, known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05


def test_two_sample_crossover():
    """ Tests for hypotheses on two sample means crossover design """

    # From Chow et al 3.3.4
    h = TwoSampleCrossover(mu_1=10, mu_2=0, stdev=2*20, alpha=0.05,
                           power=0.9, hypothesis="equality",
                           known_stdev=False)
    h.calculate()
    assert h.n == 86
    assert h.power > 0.80

    h = TwoSampleCrossover(mu_1=-0.1, mu_2=0, stdev=0.2, alpha=0.05,
                           power=0.8, hypothesis="superiority",
                           margin=0.2)
    h.calculate()
    assert h.n == 13
    assert h.power > 0.80

    h = TwoSampleCrossover(mu_1=-0.1, mu_2=0, stdev=0.2, n=13,
                           power=0.8, hypothesis="superiority",
                           margin=0.2)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleCrossover(mu_1=-0.1, mu_2=0, stdev=0.2, alpha=0.05,
                           n=13, hypothesis="superiority",
                           margin=0.2)
    h.calculate()
    assert h.power > 0.80

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, alpha=0.05,
                           power=0.8, hypothesis="superiority",
                           margin=0.2, known_stdev=False)
    h.calculate()
    assert h.n == 14
    assert h.power > 0.80

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, n=14,
                           power=0.8, hypothesis="superiority",
                           margin=0.2, known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, alpha=0.05,
                           n=14, hypothesis="superiority",
                           margin=0.2, known_stdev=False)
    h.calculate()
    assert h.n == 14
    assert h.power > 0.80

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, alpha=0.05,
                           power=0.8, hypothesis="equivalence",
                           margin=0.25)
    h.calculate()
    assert h.n == 8
    assert h.power > 0.80

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, n=8,
                           power=0.8, hypothesis="equivalence",
                           margin=0.25)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, n=8,
                           alpha=0.05, hypothesis="equivalence",
                           margin=0.25)
    h.calculate()
    assert h.power > 0.80

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, alpha=0.05,
                           power=0.8, hypothesis="equivalence",
                           margin=0.25, known_stdev=False)
    h.calculate()
    assert h.n == 9
    assert h.power > 0.80

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, n=9,
                           power=0.8, hypothesis="equivalence",
                           margin=0.25, known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleCrossover(mu_1=0.9, mu_2=1, stdev=0.2, n=9,
                           alpha=0.05, hypothesis="equivalence",
                           margin=0.25, known_stdev=False)
    h.calculate()
    assert h.power > 0.80

    # From the PASS 11 Manual: pg 500-10
    # Note: PASS uses a different stdev which is equal to half the stdev
    # used in these calculations.
    h = TwoSampleCrossover(mu_1=10, mu_2=0, stdev=2*20, n=86,
                           power=0.9, hypothesis="equality",
                           known_stdev=False)
    h.calculate()
    assert h.alpha < 0.05

    h = TwoSampleCrossover(mu_1=10, mu_2=0, stdev=2*10, n=40,
                           alpha=0.05, hypothesis="equality",
                           known_stdev=False)
    h.calculate()
    assert h.power >= 0.8690


def test_one_way_anova_simultaneous():
    h = OneWayAnova(mu=[8.25, 11.75, 12.00, 13.00], stdev=3.5,
                    comparison='simultaneous', alpha=0.05, power=0.8)
    h.calculate()
    assert h.n == 11
    assert h.power > 0.80

    h = OneWayAnova(n=11, mu=[8.25, 11.75, 12.00, 13.00], stdev=3.5,
                    comparison='simultaneous', alpha=0.05)
    h.calculate()
    assert h.power > 0.80

    h = OneWayAnova(n=11, mu=[8.25, 11.75, 12.00, 13.00], stdev=3.5,
                    comparison='simultaneous', power=0.8)
    h.calculate()
    assert h.alpha < 0.05


def test_multi_sample_williams():
    h = MultiSampleWilliams(mu=[0.20, 0.15, 0.25], stdev=0.1,
                            hypothesis='equality', alpha=0.05, power=0.8,
                            known_stdev=True)
    h.calculate()
    assert h.n == 6
    assert h.power > 0.80

    h = MultiSampleWilliams(n=6, mu=[0.20, 0.15, 0.25], stdev=0.1,
                            hypothesis='equality', alpha=0.05,
                            known_stdev=True)
    h.calculate()
    assert h.power > 0.80

    h = MultiSampleWilliams(n=6, mu=[0.20, 0.15, 0.25], stdev=0.1,
                            hypothesis='equality', power=0.80,
                            known_stdev=True)
    h.calculate()
    assert h.alpha < 0.05
