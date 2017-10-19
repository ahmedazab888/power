import scipy.stats as stats
import math


def calc_stdev(l, t, t_0):
    denom = 1 + (math.exp(-l * t) * (1 - math.exp(-l * t_0))) / (t_0 * l)
    stdev = math.sqrt(l**2 / denom)
    return stdev


def time_to_event(contrast=None,
                  response=None,
                  proportion=None,
                  t=None,
                  t_0=None,
                  n=None,
                  alpha=None,
                  power=None):
    # Check inputs
    if n is None:
        if alpha is None or power is None:
            raise ValueError("Two of n, alpha and power must be defined")
        unknown = "n"
        if power <= 0 or power >= 1:
            raise ValueError("Power must be in (0, 1)")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Alpha must be in (0, 1)")
    elif alpha is None:
        if power is None:
            raise ValueError("Two of n, alpha and power must be defined")
        unknown = "alpha"
        if n < 2:
            raise ValueError("N must be greater than 2")
        if power <= 0 or power >= 1:
            raise ValueError("Power must be in (0, 1)")
    elif power is None:
        if n < 2:
            raise ValueError("N must be greater than 2")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Alpha must be in (0, 1)")
        unknown = "power"
    else:
        raise ValueError("There must be at least one unknown value")

    if len(contrast) != len(response):
        raise ValueError("Contrast and response must have the same length")
    n_group = len(contrast)

    if sum(contrast) != 0:
        raise ValueError("Contrast must sum to 1")

    if t <= 0:
        raise ValueError("Trial time must be positive")

    if t_0 <= 0:
        raise ValueError("Accrual time must be positive")

    if proportion is None:
        proportion = [1 / n_group] * n_group
    else:
        if len(proportion) != n_group:
            raise ValueError("Contrast and proportion must have the same length")
        if sum(proportion) != 1:
            raise ValueError("Proportion must sum to 1")

    # Perform calculation
    c_by_f = sum([(c * c) / f for c, f in zip(contrast, proportion)])
    epsilon = sum([c * m for c, m in zip(contrast, response)])
    distribution = stats.norm()
    stdev = [calc_stdev(l, t, t_0) for l in response]
    mean_stdev = sum([s * f for s, f in zip(stdev, proportion)])

    alpha_factor = mean_stdev * math.sqrt(c_by_f)
    beta_factor = sum([(c * c) / f * s for c, f, s in zip(contrast, proportion, stdev)])
    beta_factor = math.sqrt(beta_factor)

    if unknown == "n":
        z_alpha = distribution.ppf(1 - alpha)
        z_beta = distribution.ppf(power)

        n = ((z_alpha * alpha_factor + z_beta * beta_factor) / epsilon)**2
        n = math.ceil(n)
        z_beta = math.sqrt(n / c_by_f) * epsilon - z_alpha * alpha_factor
        z_beta /= beta_factor
        power = stats.norm.cdf(z_beta)
    elif unknown == "alpha":
        z_beta = distribution.ppf(power)
        z_alpha = math.sqrt(n / c_by_f) * epsilon - z_beta * beta_factor
        z_alpha /= alpha_factor
        alpha = 1 - stats.norm.cdf(z_alpha)
    elif unknown == "power":
        z_alpha = distribution.ppf(1 - alpha)
        z_beta = math.sqrt(n / c_by_f) * epsilon - z_alpha * alpha_factor
        z_beta /= beta_factor
        power = stats.norm.cdf(z_beta)

    return n, power, alpha
