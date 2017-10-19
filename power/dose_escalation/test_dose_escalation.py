""" Test Cases for Linear Contrasts module
"""

import pytest
import power.dose_escalation as de


def test_continuous_linear_contrasts():
    """ Test Cases for linear contrats """

    contrast = [-6, 1, 2, 3]
    bad_contrast = [-3, -1, 0, 3]

    response = [0.05, 0.12, 0.14, 0.16]

    # Error Conditions
    # sum(contrasts) must equal 0
    with pytest.raises(ValueError):
        de.continuous(contrast=bad_contrast,
                      response=response,
                      stdev=0.22,
                      alpha=0.05,
                      power=0.8)

    # stdev must be positive
    with pytest.raises(ValueError):
        de.continuous(contrast=contrast,
                      response=response,
                      stdev=-0.22,
                      alpha=0.05,
                      power=0.8)

    # alpha must be in (0, 1)
    with pytest.raises(ValueError):
        de.continuous(contrast=contrast,
                      response=response,
                      stdev=0.22,
                      alpha=-0.05,
                      power=0.8)

    with pytest.raises(ValueError):
        de.continuous(contrast=contrast,
                      response=response,
                      stdev=0.22,
                      alpha=1.05,
                      power=0.8)

    # power must be in (0, 1)
    with pytest.raises(ValueError):
        de.continuous(contrast=contrast,
                      response=response,
                      stdev=0.22,
                      alpha=0.05,
                      power=-0.8)

    with pytest.raises(ValueError):
        de.continuous(contrast=contrast,
                      response=response,
                      stdev=0.22,
                      alpha=0.05,
                      power=1.8)

    # Base calculation
    n, power, alpha = de.continuous(contrast=contrast,
                                    response=response,
                                    stdev=0.22,
                                    alpha=0.05,
                                    power=0.8)
    assert n == 178
    assert power > 0.8
    assert alpha == 0.05

    n, power, alpha = de.continuous(contrast=contrast,
                                    response=response,
                                    stdev=0.22,
                                    alpha=0.05,
                                    n=178)
    assert n == 178
    assert power > 0.8
    assert alpha == 0.05

    n, power, alpha = de.continuous(contrast=contrast,
                                    response=response,
                                    stdev=0.22,
                                    power=0.8,
                                    n=178)
    assert n == 178
    assert power >= 0.8
    assert alpha == 0.05


def test_binary_response():
    contrast = [-3, -1, 1, 3]

    response = [0.1, 0.3, 0.5, 0.7]

    n, power, alpha = de.binary(contrast=contrast,
                                response=response,
                                power=0.8,
                                alpha=0.05)
    # Chow has 26, we get 27 due to rounding (n=26.00788)
    print(power)
    print(alpha)
    assert n == 27
    assert power >= 0.8
    assert alpha <= 0.05


def test_time_to_event_response():
    contrast = [-6, 1, 2, 3]
    response = [0.0495, 0.0347, 0.0315, 0.0289]

    n, power, alpha = de.time_to_event(contrast=contrast,
                                       response=response,
                                       t=16,
                                       t_0=9,
                                       power=0.8,
                                       alpha=0.05)
    assert n == 666
    assert power >= 0.8
    assert alpha <= 0.05
