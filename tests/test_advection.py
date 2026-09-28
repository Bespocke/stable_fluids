import numpy as np

from fluid2d.solver.numba_mod import advect_numba
from fluid2d.solver.solid import advect_numba_solid


def test_zero_velocity_preserves_scalar():
    rng = np.random.default_rng(123)
    source = rng.random((32, 32), dtype=np.float32)
    dest = np.zeros_like(source)
    vx = np.zeros_like(source)
    vy = np.zeros_like(source)

    advect_numba(source, dest, vx, vy, 1.0, 0.1)

    np.testing.assert_allclose(dest[1:-1, 1:-1], source[1:-1, 1:-1], atol=1e-6)


def test_constant_scalar_is_preserved_with_solids():
    source = np.full((32, 32), 2.5, dtype=np.float32)
    dest = np.zeros_like(source)
    vx = np.full_like(source, 0.2)
    vy = np.full_like(source, -0.1)
    solid = np.zeros_like(source, dtype=np.bool_)
    solid[14:18, 14:18] = True

    advect_numba_solid(source, dest, vx, vy, 1.0, 0.1, solid)

    np.testing.assert_allclose(dest[1:-1, 1:-1][~solid[1:-1, 1:-1]], 2.5, atol=1e-6)
    np.testing.assert_allclose(dest[solid], 2.5, atol=1e-6)
