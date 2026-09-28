# Changelog

## 0.1.0 - first public release candidate

- Clean package layout with `pyproject.toml`.
- Numba-accelerated 2D incompressible Stable Fluids solver.
- Smagorinsky eddy-viscosity model.
- Vorticity confinement.
- Rasterized solid obstacles and boundary-aware advection/diffusion helpers.
- External/body force support.
- Smoke/density and temperature passive scalars.
- Pygame visualization.
- Basic unit tests and validation roadmap.
- Fixed a solid-aware vorticity bug where `dv/dy` was accidentally used instead of `du/dy`.
- Fixed rendering axis convention so row/column physics arrays map consistently to Pygame x/y coordinates.
