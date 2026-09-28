"""Interactive smoke demo: a driven jet interacting with a rasterized cylinder."""

import numpy as np
from numba import set_num_threads

from fluid2d import Fluid, step
from fluid2d.render.renderer import Renderer


def main() -> None:
    n = 256
    scale = 3
    dx = 1.0
    dt = 0.1

    # Optional: tune this on your machine.
    set_num_threads(7)

    fluid = Fluid(n, dx, dt)
    fluid.viscosity = 0.1
    fluid.density_diffusion = 0.0

    y, x = np.mgrid[0:n, 0:n]
    cx = cy = n // 2
    fluid.solid_mask[(x - cx) ** 2 + (y - cy) ** 2 < 15 ** 2] = True

    jet = (slice(10, 20), slice(n // 2 - 110, n // 2 + 110))

    fluid.vx[jet] = 5.0

    renderer = Renderer(n, scale)
    running = True

    while running:
        for event in renderer.events():
            if event.type == renderer.quit_event:
                running = False

        # Re-inject the driven smoke source every frame.
        fluid.density[jet] = 1.0
        fluid.vy[jet] = 30.0
        fluid.vx[jet] = 0.0

        step(fluid, solve_density=True)
        renderer.render_density(fluid)
        renderer.tick(60)

    renderer.close()


if __name__ == "__main__":
    main()
