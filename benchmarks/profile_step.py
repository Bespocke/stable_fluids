"""Minimal benchmark skeleton for timing one solver step."""

import time

import numpy as np

from fluid2d import Fluid, step


def main() -> None:
    n = 256
    fluid = Fluid(n, 1.0, 0.1)
    fluid.viscosity = 0.1
    fluid.density_diffusion = 0.0

    y, x = np.mgrid[0:n, 0:n]
    fluid.solid_mask[(x - n // 2) ** 2 + (y - n // 2) ** 2 < 15 ** 2] = True
    fluid.density[10:20, 20:-20] = 1.0
    fluid.vy[10:20, 20:-20] = 30.0

    # Warm up Numba compilation.
    step(fluid, iterations=50, solve_density=True)

    repeats = 20
    start = time.perf_counter()
    for _ in range(repeats):
        step(fluid, iterations=50, solve_density=True)
    elapsed = time.perf_counter() - start

    print(f"Grid: {n}x{n}")
    print(f"Average step: {elapsed / repeats * 1e3:.2f} ms")
    print(f"Approx. FPS: {repeats / elapsed:.1f}")


if __name__ == "__main__":
    main()
