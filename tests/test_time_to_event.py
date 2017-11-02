# from skdesign.power.time_to_event import (Cox,
#                                           Exponential)


# def test_cox():
#     """ Test Cases for Cox Proportional Hazards Model """

#     # From Chow et al. 7.3.4
#     h = Cox(alpha=0.05, power=0.8, hypothesis='equality',
#             hazard_ratio=2, control_proportion=0.5,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.n == 82
#     assert h.power > 0.8

#     h = Cox(alpha=0.05, n=82, hypothesis='equality',
#             hazard_ratio=2, control_proportion=0.5,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.power > 0.8

#     h = Cox(n=82, power=0.8, hypothesis='equality',
#             hazard_ratio=2, control_proportion=0.5,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.alpha < 0.05

#     h = Cox(alpha=0.05, power=0.8, hypothesis='superiority',
#             hazard_ratio=2, control_proportion=0.5, margin=0.3,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.n == 200
#     assert h.power > 0.8

#     h = Cox(alpha=0.05, n=200, hypothesis='superiority',
#             hazard_ratio=2, control_proportion=0.5, margin=0.3,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.power > 0.8

#     h = Cox(n=200, power=0.8, hypothesis='superiority',
#             hazard_ratio=2, control_proportion=0.5, margin=0.3,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.alpha < 0.05

#     # Chow has 171.  We get 172.  The difference is due to rounding.
#     h = Cox(alpha=0.05, power=0.8, hypothesis='equivalence',
#             hazard_ratio=1, control_proportion=0.5, margin=0.5,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.n == 172
#     assert h.power > 0.8

#     # Chow has 171.  We get 172.  The difference is due to rounding.
#     h = Cox(alpha=0.05, n=172, hypothesis='equivalence',
#             hazard_ratio=1, control_proportion=0.5, margin=0.5,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.power > 0.8

#     # Chow has 171.  We get 172.  The difference is due to rounding.
#     h = Cox(n=172, power=0.8, hypothesis='equivalence',
#             hazard_ratio=1, control_proportion=0.5, margin=0.5,
#             treatment_proportion=0.5, proportion_visible=0.8)
#     h.calculate()
#     assert h.alpha < 0.05


# def test_exponential():
#     """ See Chow et al. 7.2.4.  Note: Chow has an error in his calculations:
#     for the second factor in the calculation of n, he uses the sum of the
#     variances, not the inverse of the sum.  Hence a descrepancy.

#     These should be double checked, but where as Chow has 39, 48, and 67,
#     we get 41, 50, and 75.  This is consistant with Chow's error
#     """

#     h = Exponential(alpha=0.05, power=0.8, hypothesis='equality',
#                     control_hazard=2, treatment_hazard=1, gamma=0,
#                     trial_time=3, accrual_time=1)
#     h.calculate()
#     assert h.n_1 == 41
#     assert h.n_2 == 41
#     assert h.power > 0.8

#     h = Exponential(alpha=0.05, power=0.8, margin=0.2,
#                     hypothesis='superiority', control_hazard=2,
#                     treatment_hazard=1, gamma=0,
#                     trial_time=3, accrual_time=1)
#     h.calculate()
#     assert h.n_1 == 50
#     assert h.n_2 == 50
#     assert h.power > 0.8

#     h = Exponential(ratio=1, alpha=0.05, power=0.8, margin=0.5,
#                     hypothesis='equivalence', control_hazard=1,
#                     treatment_hazard=1, gamma=0,
#                     trial_time=3, accrual_time=1)
#     h.calculate()
#     assert h.n_1 == 75
#     assert h.n_2 == 75
#     assert h.power > 0.8
