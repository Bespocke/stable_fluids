# Validation plan

This project is a numerical/educational prototype, so validation is split into three layers.

## Numerical checks

- Constant scalar under zero velocity.
- Constant scalar in the presence of rasterized solids.
- Known analytical derivatives for gradients and vorticity.
- Divergence reduction for a projection step without solids.

## Physical cases to add

1. Poiseuille flow.
2. Couette flow.
3. Taylor-Green vortex.
4. Lid-driven cavity.
5. Flow around a cylinder.
6. Vortex decay.
7. Kelvin-Helmholtz instability.

For each case, report an error norm, a conserved quantity, or a reference quantity whenever a reference solution is available.

## Important limitation of V0.1

The current solver uses a cell-centered velocity/pressure discretization with rasterized internal solids. Its pressure projection near arbitrary solid interfaces is intentionally kept simple. This can produce spurious near-wall behavior (including apparent suction/attraction) and is the main numerical limitation to address before claiming engineering-level force predictions.

A MAC-grid formulation with a boundary-consistent projection is planned as a future solver backend rather than being silently mixed into this V0.1 implementation.
