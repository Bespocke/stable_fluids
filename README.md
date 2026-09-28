# Fluid2D — lightweight 2D incompressible flow solver

A small Python/Numba solver for interactive 2D incompressible fluid simulation.

The project is intentionally positioned between a teaching tool and a lightweight prototyping engine: the numerical methods stay visible and understandable while hot loops are accelerated with Numba.

> **Status:** early public release (v0.1.0). This is not an engineering-certified CFD package. Near-wall flow around rasterized internal solids is the main known numerical limitation.

## Current model

- Incompressible Navier–Stokes on a uniform cell-centered Cartesian grid.
- Semi-Lagrangian advection.
- Jacobi iterations for diffusion and pressure Poisson.
- Smagorinsky eddy viscosity for the velocity/scalar diffusion operator.
- Vorticity confinement.
- Fixed rasterized solid obstacles.
- Constant body-force vector support.
- Passive density and temperature fields.
- Pygame visualization.
- `float32` state arrays and reusable work buffers.
- Numba JIT + optional parallel loops.

The implementation deliberately favors clarity and interactive robustness over strict conservative CFD discretization.

## Installation

Create an environment and install the core package:

```bash
python -m pip install -e .
```

For the Pygame demonstrations:

```bash
python -m pip install -e '.[viz]'
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
```

## Run the tests

```bash
pytest
```

The first run includes Numba compilation overhead; subsequent runs are faster.

## Examples

From the repository root:

```bash
PYTHONPATH=src python examples/cylinder_smoke.py
PYTHONPATH=src python examples/kelvin_helmholtz.py
```

The examples are deliberately simple so that the numerical pipeline remains easy to inspect.

## Solver pipeline

A timestep currently follows the structure:

```text
velocity diffusion
        ↓
vorticity confinement
        ↓
body forces
        ↓
pressure projection
        ↓
vorticity / strain diagnostics
        ↓
velocity advection
        ↓
pressure projection
        ↓
scalar advection
        ↓
scalar diffusion
```

The pressure projection solves a Poisson equation with Jacobi iterations.

## State layout

The V0.1 solver is cell-centered:

```text
vx, vy, p, density, temperature
                ↓
           shape (N, N)
```

Work arrays are allocated once in `Fluid` and reused between steps.

## Solid boundaries

Solids are represented by a boolean raster mask. The current implementation uses ghost-like values for local operators and special treatment during scalar/velocity advection and diffusion.

This is deliberately a V0 approach. The main known limitation is the pressure/velocity coupling near arbitrary internal solid interfaces: because the solver is cell-centered, the wall condition is only approximated on a rasterized boundary. This can produce spurious near-wall forces or apparent suction.

A future MAC-grid backend is planned so that normal fluxes live directly on cell faces.

## Physics roadmap

### V0.1

- Stable Fluids core.
- Smagorinsky LES-style viscosity.
- Solids, forces, density and temperature.
- Tests, diagnostics, examples and documentation.

### V0.x

- Better Poisson solver (Gauss–Seidel / SOR / multigrid).
- More rigorous solid boundary treatment.
- Physical-unit configuration and validation suite.
- Scene files and a scene editor.
- Python API for reusable simulations.

### Later

- MAC-grid incompressible solver.
- More conservative advection.
- RANS and compressible solvers.
- C++ core and GPU backends.

## Validation

See [`docs/VALIDATION.md`](docs/VALIDATION.md). Planned reference cases include Poiseuille, Couette, Taylor–Green vortex, lid-driven cavity, vortex decay and flow around a cylinder.

## Design goal

The long-term goal is a lightweight, intuitive and open simulator for learning and rapid flow prototyping. The project should expose the physics instead of hiding it behind a large configuration system, while keeping numerical methods and their limits explicit.

## Development

AI tools were used as development assistants for parts of the project, notably for debugging, code review, documentation, and GitHub/deployment setup, as software development is not my main field of study or area of specialization.

The numerical methods, solver architecture, implementation, experiments, and validation were designed and carried out by me.

The development history, source code, and intermediate versions of the project can be provided upon request. Feel free to contact me if further details are needed.

## License

MIT. Change the license before publishing if you prefer another licensing model.
