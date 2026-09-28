# Solid interaction: current diagnosis

## Confirmed bug fixed in 0.1.0

The solid-aware vorticity operator had this structure:

```text
dvy/dx - d(vy)/dy
```

because the south/north terms of the second derivative accidentally read the `vy` field instead of `vx`.

The intended 2D scalar vorticity is:

```text
omega = dvy/dx - du/dy
```

The public version fixes that indexing error. A regression test (`tests/test_vorticity.py`) uses `u=-y`, `v=x`, whose exact curl is `2`.

## Likely source of the remaining apparent suction

The current internal-solid projection is cell-centered. Three choices interact:

1. `divergence_solid()` replaces the normal velocity next to a solid by zero but still uses a centered `1/(2 dx)` stencil.
2. The pressure solve removes solid neighbors from the stencil and divides by the remaining neighbor count.
3. `annex()` forces velocities to zero in every cell touching a solid before/after the pressure-gradient correction.

Those three operations are not a single consistent finite-volume / ghost-cell discretization of the wall boundary. In particular, the physical no-penetration condition belongs at the wall, not at the center of the entire neighboring fluid cell. This can create spurious pressure/velocity patterns near rasterized obstacles and can look like an obstacle is attracting fluid.

The same issue is visible in a controlled projection diagnostic: a velocity field with a solid wall is not driven to the same low divergence obtained on the no-solid domain.

## Scalar advection limitation

The current scalar advection already avoids directly reading solid cells by replacing sampled solid corners with the current cell value. This approximates a zero-normal-gradient condition, but it does not yet clip the complete semi-Lagrangian backtrace at the solid/fluid interface. A backtrace can therefore still cross a rasterized obstacle in a way that is not geometrically exact.

## Why this is not patched blindly

A robust correction should change the discrete boundary formulation as a whole rather than add another `if solid:` after the existing operators. The clean long-term solutions are:

- a consistent ghost-cell/cut-cell cell-centered projection, or
- a MAC grid where normal velocities live directly on cell faces.

The current V0.1 therefore fixes the confirmed curl error and documents the near-wall projection limitation instead of claiming that the existing cell-centered treatment is already quantitatively correct.
