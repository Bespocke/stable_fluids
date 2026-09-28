"""Kelvin-Helmholtz-style smoke test using the current solver."""

import numpy as np
from numba import set_num_threads

from fluid2d import Fluid, step
from fluid2d.render.renderer import Renderer


def main() -> None:
    n = 256
    set_num_threads(7)
    fluid = Fluid(n, 1.0, 0.1)
    fluid.viscosity = 1e-4
    fluid.density_diffusion = 1e-6

    fluid.density[:, : n // 2] = 0.8
    fluid.density[:, n // 2 :] = 0.5
    fluid.vy[:, : n // 2] = 20.0
    fluid.vy[:, n // 2 :] = -20.0

    renderer = Renderer(n, 3)
    running = True
    while running:
        for event in renderer.events():
            if event.type == renderer.quit_event:
                running = False
        step(fluid, solve_density=True)
        renderer.render_density(fluid)
        renderer.tick(60)
    renderer.close()


if __name__ == "__main__":
    main()
