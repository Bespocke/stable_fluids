import numpy as np

from fluid2d.boundaries.operators import rotationel_solid


def test_rotationel_solid_uses_du_dy_component():
    n = 32
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    # vx = -y, vy = x -> omega_z = dvy/dx - du/dy = 1 - (-1) = 2
    vx = -y
    vy = x
    curl = np.zeros_like(vx)
    solid = np.zeros_like(vx, dtype=np.bool_)

    rotationel_solid(vx, vy, curl, 1.0, solid)

    np.testing.assert_allclose(curl[2:-2, 2:-2], 2.0, atol=1e-6)
