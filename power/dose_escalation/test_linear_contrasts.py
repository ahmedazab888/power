""" Test Cases for Linear Contrasts module
"""

import pytest
import power.dose_escalation as de


def test_linear_contrasts():
    """ Test Cases for linear contrats """

    contrast = [-6, 1, 2, 3]
    linear_contrast = [-3, -1, 1, 3]
    step_contrast = [-3, 0, 0, 3]
    umbrella_contrast = [-3.25, -0.25, 2.75, 0.75]
    convex_contrast = [-1.25, -1.25, -1.25, 3.75]
    concave_contrast = [-3.75, 1.25, 1.25, 1.25]
    bad_contrast = [-3, -1, 0, 3]

    response = [0.05, 0.12, 0.14, 0.16]
    linear_response = [0.1, 0.3, 0.5, 0.7]
    step_response = [0.1, 0.4, 0.4, 0.7]
    umbrella_response = [0.1, 0.4, 0.7, 0.5]
    convex_response = [0.1, 0.1, 0.1, 0.6]
    concave_response = [0.1, 0.6, 0.6, 0.6]

    # Error Conditions
    # sum(contrasts) must equal 0
    with pytest.raises(ValueError):
        de.linear_contrasts(contrast=bad_contrast,
                            response=response,
                            stdev=0.22,
                            alpha=0.05,
                            power=0.8)

    # stdev must be positive
    with pytest.raises(ValueError):
        de.linear_contrasts(contrast=contrast,
                            response=response,
                            stdev=-0.22,
                            alpha=0.05,
                            power=0.8)

    # alpha must be in (0, 1)
    with pytest.raises(ValueError):
        de.linear_contrasts(contrast=contrast,
                            response=response,
                            stdev=0.22,
                            alpha=-0.05,
                            power=0.8)
    with pytest.raises(ValueError):
        de.linear_contrasts(contrast=contrast,
                            response=response,
                            stdev=0.22,
                            alpha=1.05,
                            power=0.8)

    # power must be in (0, 1)
    with pytest.raises(ValueError):
        de.linear_contrasts(contrast=contrast,
                            response=response,
                            stdev=0.22,
                            alpha=0.05,
                            power=-0.8)

    with pytest.raises(ValueError):
        de.linear_contrasts(contrast=contrast,
                            response=response,
                            stdev=0.22,
                            alpha=0.05,
                            power=1.8)

    Base calculation
    n, power, alpha = de.linear_contrasts(contrast=contrast,
                                          response=response,
                                          stdev=0.22,
                                          alpha=0.05,
                                          power=0.8)
    assert n == 178
    assert power > 0.8
    assert alpha == 0.05

    n, power, alpha = de.linear_contrasts(contrast=contrast,
                                          response=response,
                                          stdev=0.22,
                                          alpha=0.05,
                                          n=178)
    assert n == 178
    assert power > 0.8
    assert alpha == 0.05

    n, power, alpha = de.linear_contrasts(contrast=contrast,
                                          response=response,
                                          stdev=0.22,
                                          power=0.8,
                                          n=178)
    assert n == 178
    assert power == 0.8
    assert alpha < 0.05
