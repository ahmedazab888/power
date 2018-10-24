# """ Test cases for the power.gof module """

# # import pytest
# from power.gof import (Pearson,
#                        PearsonIndependance,
#                        CMH,
#                        StuartMaxwell,
#                        McNemar,
#                        CarryOverEffect)


# def test_pearson():
#     """ From Chow et al. 6.1.2 """
#     p = [0.2, 0.6, 0.2]
#     p_0 = [0.25, 0.45, 0.30]
#     h = Pearson(p=p, p_0=p_0, alpha=0.05, power=0.8)
#     h.calculate()
#     assert h.n == 104


# def test_pearson_independance():
#     """ See Chow 6.2.3 """
#     p = [[2.0/20.0, 7.0/20.0, 1.0/20.0],
#          [2.0/20.0, 5.0/20.0, 3.0/20.0]]
#     h = PearsonIndependance(p=p, alpha=0.05, power=0.8)
#     h.calculate()
#     assert h.n == 145


# def test_cmh():
#     """ See Chow et al. 6.3.2 """
#     p = [[[0.35, 0.15], [0.25, 0.25]],
#          [[0.30, 0.20], [0.20, 0.30]],
#          [[0.40, 0.10], [0.20, 0.30]],
#          [[0.35, 0.15], [0.15, 0.35]]]
#     h = CMH(p=p, alpha=0.05, power=0.8)
#     h.calculate()
#     assert h.n == 86
#     assert h.power > 0.8

#     h = CMH(p=p, alpha=0.05, n=86)
#     h.calculate()
#     assert h.power > 0.8

#     h = CMH(p=p, n=86, power=0.8)
#     h.calculate()
#     assert h.alpha < 0.05


# def test_stuart_maxwell():
#     """ See Chow et al. 6.2.3

#     Chow has 102, but we get 103 due to roudning """
#     p = [[3.0/25.0, 4.0/25.0, 4.0/25.0],
#          [2.0/25.0, 5.0/25.0, 3.0/25.0],
#          [1.0/25.0, 2.0/25.0, 3.0/25.0]]
#     h = StuartMaxwell(p=p, alpha=0.05, power=0.8)
#     h.calculate()
#     assert h.n == 103


# def test_mcnemar():
#     """ See Chow et al. 6.4.3 """
#     h = McNemar(p_01=0.2, p_10=0.50, alpha=0.05, power=0.8)
#     h.calculate()
#     assert h.n == 59
#     assert h.power > 0.8

#     h = McNemar(p_01=0.2, p_10=0.50, alpha=0.05, n=59)
#     h.calculate()
#     assert h.power > 0.8

#     h = McNemar(p_01=0.2, p_10=0.50, n=59, power=0.8)
#     h.calculate()
#     assert h.alpha < 0.05


# def test_carry_over_effect():
#     """ See Chow et al. 6.5.2 """
#     h = CarryOverEffect(gamma=0.89, stdev_1=2.3, stdev_2=2.4,
#                         alpha=0.05, power=0.8)
#     h.calculate()
#     assert h.n == 110
#     assert h.power > 0.8

#     h = CarryOverEffect(gamma=0.89, stdev_1=2.3, stdev_2=2.4,
#                         alpha=0.05, n=110)
#     h.calculate()
#     assert h.power > 0.8

#     h = CarryOverEffect(gamma=0.89, stdev_1=2.3, stdev_2=2.4,
#                         n=110, power=0.8)
#     h.calculate()
#     assert h.alpha < 0.05
