from skdesign.group_sequential import pocock_cutoff
import pytest


def test_pocock():
    c = pocock_cutoff(5, 0.01)
    assert c == 2.986

    c = pocock_cutoff(3, 0.10)
    assert c == 1.992

    with pytest.raises(ValueError):
        pocock_cutoff(0.01, 5)

    with pytest.raises(ValueError):
        pocock_cutoff(11, 0.05)

    with pytest.raises(ValueError):
        pocock_cutoff(0, 0.05)

    with pytest.raises(ValueError):
        pocock_cutoff(5, 0.101)

    with pytest.raises(ValueError):
        pocock_cutoff(5, 0.11)
