from power.bioequivalence import (Average,
                                  Individual,
                                  InVitro,
                                  Population)
import math


def test_average_bioequivalence():
    """ Tests for Average Bioequivalence. """

    # See 10.2 Example from Chow et al.
    h = Average(delta=0.223, stdev=0.40, margin=0.05,
                alpha=0.05, power=0.8, known_stdev=True)
    h.calculate()
    # Chow has 21, but they have the wrong z_beta/2.  It should be 1.28,
    # not 0.84.  When that is fixed, the correct n is 23
    assert h.n == 23
    assert h.power > 0.8


def test_population_bioequivalence():
    """ Tests for Population Bioequivalence. """

    # See 10.3 Example from Chow et al.
    h = Population(l=-0.2966, stdev_11=0.2, stdev_tt=math.sqrt(0.17),
                   stdev_tr=math.sqrt(0.17), stdev_bt=0.4, stdev_br=0.4,
                   rho=0.75, alpha=0.05, power=0.8)
    h.calculate()
    assert h.n == 12


# def test_individual_bioequivalence():
#     """ Tests for Population Bioequivalence. """
#     # See 10.4 Example from Chow et al.
#     h = Individual(delta=0, stdev_wr=0.4, stdev_wt=0.6, rho=0.75,
#                    stdev_br=0.4, stdev_bt=0.1, alpha=0.05, power=0.8)
#     h.calculate()
#     assert h.n == 22


# def test_in_vitro_bioequivalence():
#     """ Tests for Population Bioequivalence. """
#     # See 10.4 Example from Chow et al.
#     h = InVitro(delta=0, stdev_wr=0.5, stdev_wt=0.5, stdev_br=0.5,
#                 stdev_bt=0.5, alpha=0.05, power=0.8, )
#     h.calculate()
#     assert h.n == 1
#     assert h.m == 47
