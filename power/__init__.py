import numbers


class PowerBase(object):
    """ The base for specific power objects.

    The PowerBase class is an class that holds certain, common elements
    that all hypotheses share such as :math:`\\alpha` and power.  It
    also moves a lot of the value checking from the function that calculates
    the power of the hypothesis to a single place.

    The classes that inherit from PowerBase are for specific power
    calculations, such as a hypothesis on means.

    Attributes:
        alpha: The :math:`\\alpha` level required by the hypothesis.
        beta: The :math:`\\beta` level required by the hypothesis (equal to
            :math:`1` - power).
        power: The power required by the hypothesis (equal to
            :math:`1 - \\beta`).
        hypothesis: Either ['equality', 'superiority', 'equivalence'].  If
            a hypothesis does not have those selections, it will pass the value
            'equality' in it's __init__ function.
        _alpha_adjustment: An adjustment used in several calculations dependant
            on the hypothesis.
        _beta_adjustment: An adjustment used in several calculations dependant
            on the hypothesis.
    """

    def __init__(self, alpha=None, power=None, beta=None, hypothesis=None):
        """ Initialize a Hypothesis object.
        """

        # Check value of and assign alpha.  Alpha must be in [0, 1].
        if alpha is not None:
            is_in_0_1(alpha, 'alpha')
            self.alpha = alpha
        else:
            self.alpha = None

        # Check values of and assign power and beta.  Since beta = 1 - power,
        # the two values have to be handled together.  The below code goes
        # through the possible comibinations of power and beta being assigned
        # and ensures they hold proper values (i.e. in [0, 1] and
        # `beta` = 1 - `power` ).
        if power is not None:
            is_in_0_1(power, 'power')
            if beta is not None:
                if beta is not (1 - power):
                    print(beta)
                    print(power)
                    raise ValueError("`power` does not equal 1 - `beta`.")
                else:
                    self.beta = beta
                    self.power = power
            else:
                self.power = power
                self.beta = 1 - power
        elif beta is not None:
            is_in_0_1(beta, 'beta')
            self.beta = beta
            self.power = 1 - beta
        else:
            self.beta = None
            self.power = None

        if hypothesis in ['equality', 'superiority', 'equivalence']:
            self.hypothesis = hypothesis
        elif hypothesis is None:
            self.hypothesis = 'equality'
        else:
            raise ValueError(("`hypothesis` should be in ['equality', "
                              "'superiority', 'equivalence']"))

        self._set_adjustments()

    def __repr__(self):
        """ The canonical representation of a Hypothesis object
        """
        representation = "Alpha: " + str(self.alpha) + "\n" + \
                         "Power: " + str(self.power) + "\n" + \
                         "Sample Size: " + str(self.n) + "\n"
        return representation

    def _set_adjustments(self):
        if self.hypothesis == 'equality':
            self._alpha_adjustment = 2
            self._beta_adjustment = 1
        elif self.hypothesis == 'superiority':
            self._alpha_adjustment = 1
            self._beta_adjustment = 1
        elif self.hypothesis == 'equivalence':
            self._alpha_adjustment = 1
            self._beta_adjustment = 2

    def _set_default_alpha(self):
        if self.alpha is None:
            self.alpha = 0.05

    def _set_default_power(self):
        if self.power is None:
            self.power = 0.8
            self.beta = 0.2


def is_in_0_1(value, value_label):
    """ Checks if a value is within [0, 1]

    Arguments:
        value: the value to check
        value_label: a name for the value to check

    Raises:
        ValueError: if `value` is not a number or not in [0, 1]
    """
    if not isinstance(value, numbers.Number):
        raise ValueError("`" + value_label + "` must be a number")
    if value > 1 or value < 0:
        raise ValueError("`" + value_label + "` must be in [0, 1]")


def is_non_negative(value, value_label):
    """ Checks if a value is non-negative

    Arguments:
        value: the value to check
        value_label: a name for the value to check

    Raises:
        ValueError: if `value` is not a number or not non-negative
    """
    if not isinstance(value, numbers.Number):
        raise ValueError("`" + value_label + "` must be a number")
    if value < 0:
        raise ValueError("`" + value_label + "` must be non-negative")


def is_positive(value, value_label):
    """ Checks if a value is postive

    Arguments:
        value: the value to check
        value_label: a name for the value to check

    Raises:
        ValueError: if `value` is not a number or not positive
    """
    if not isinstance(value, numbers.Number):
        raise ValueError("`" + value_label + "` must be a number")
    if value <= 0:
        raise ValueError("`" + value_label + "` must be postive")


def is_numeric(value, value_label):
    """ Checks if a value is a number (float or integer)

    Arguments:
        value: the value to check
        value_label: a name for the value to check

    Raises:
        ValueError: if `value` is not a number
    """
    if not isinstance(value, numbers.Number):
        raise ValueError("`" + value_label + "` must be a number")


def is_float(value, value_label):
    """ Checks if a value is a float

    Arguments:
        value: the value to check
        value_label: a name for the value to check

    Raises:
        ValueError: if `value` is not a float
    """
    if not isinstance(value, float):
        raise ValueError("`" + value_label + "` must be a float")


def is_integer(value, value_label):
    """ Checks if a value is an integer

    Arguments:
        value: the value to check
        value_label: a name for the value to check

    Raises:
        ValueError: if `value` is not a integer
    """
    if not isinstance(value, int):
        raise ValueError("`" + value_label + "` must be an integer")


def is_boolean(value, value_label):
    """ Checks if a value is a Boolean

    Arguments:
        value: the value to check
        value_label: a name for the value to check

    Raises:
        ValueError: if `value` is not a Boolean
    """
    if not isinstance(value, bool):
        raise ValueError("`" + value_label + "` must be an Boolean")
