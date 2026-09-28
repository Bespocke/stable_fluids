import numpy as np

from fluid2d.diagnostics import mass_total, relative_error


def test_mass_excludes_solids():
    density = np.ones((8, 8), dtype=np.float32)
    solid = np.zeros((8, 8), dtype=np.bool_)
    solid[:2, :2] = True
    assert mass_total(density, solid, 0.5) == 15.0


def test_relative_error():
    assert np.isclose(relative_error(1.1, 1.0), 0.1)
    assert relative_error(0.0, 0.0) == 0.0
