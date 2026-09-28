from __future__ import annotations

import numpy as np

from ..physics.vorticity_confinement import vorticity_confinment_numba
# from vorticity_confinment_numba import vorticity_confinment
from ..boundaries.operators import rotationel_solid
from .solid import advect_numba_solid, advect_speed_numba_solid, project_numba_solid, diffuse_smago_solid, diffuse_smago_speed_solid
from ..physics.smagorinsky import norm_strain

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fluid_numba_buff import Fluid

def curl_step(fluid: Fluid):

    rotationel_solid (
        fluid.vx,
        fluid.vy,
        fluid.curl,
        fluid.dx,
        fluid.solid_mask
    )

    norm_strain(
            fluid.vx,
            fluid.vy,
            fluid.gradx_tmp, 
            fluid.grady_tmp, 
            fluid.gradx_tmp2, 
            fluid.grady_tmp2, 
            fluid.normS, 
            fluid.dx,
            fluid.solid_mask
        )

from ..diagnostics import mass_total
from ..physics.forces import add_forces

def velocity_step(fluid: Fluid, iterations: int):

    # print("avant diffuse")
    # mass_total(fluid.density, fluid.solid_mask, fluid.dx, fluid.m0)

    fluid.vx, fluid.vx_tmp = diffuse_smago_speed_solid(
        fluid.vx,
        fluid.vx_tmp,
        fluid.diffuse_rhs_vx,
        fluid.viscosity,
        fluid.nu,
        fluid.normS,
        fluid.alpha,
        fluid.dt,
        fluid.dx,
        iterations,
        fluid.solid_mask
    )

    fluid.vy, fluid.vy_tmp = diffuse_smago_speed_solid(
        fluid.vy,
        fluid.vy_tmp,
        fluid.diffuse_rhs_vy,
        fluid.viscosity,
        fluid.nu,
        fluid.normS,
        fluid.alpha,
        fluid.dt,
        fluid.dx,
        iterations,
        fluid.solid_mask
    )

    # print("après diffuse")
    # mass_total(fluid.density, fluid.solid_mask, fluid.dx, fluid.m0)

    vorticity_confinment_numba(fluid.vx, fluid.vy, fluid.curl, fluid.curl_abs, fluid.gradx_tmp, fluid.grady_tmp, fluid.norm_tmp, fluid.Nx, fluid.Ny, fluid.dx, fluid.dt, fluid.solid_mask, confinment_intensity=0.5)
    # vorticity_confinment(fluid, confinment_intensity=1)

    # print("après vorticity")
    # mass_total(fluid.density, fluid.solid_mask, fluid.dx, fluid.m0)

    add_forces(fluid.forces, fluid.vx, fluid.vy, fluid.dt, fluid.solid_mask, fluid.density)

    # abs_div = np.abs(fluid.div_tmp.copy())
    # abs_div[fluid.solid_mask] = 0

    # idx = np.unravel_index(np.argmax(abs_div), abs_div.shape)
    # print("avant première proj")
    # print("max divergence:", abs_div[idx])
    # print("position:", idx)
    
    project_numba_solid(
        fluid,
        iterations
    )

    # abs_div = np.abs(fluid.div_tmp.copy())
    # abs_div[fluid.solid_mask] = 0

    # idx = np.unravel_index(np.argmax(abs_div), abs_div.shape)
    # print("première proj")
    # print("max divergence:", abs_div[idx])
    # print("position:", idx)

    curl_step(fluid)

    advect_speed_numba_solid(
        fluid.vx,
        fluid.vx_tmp,
        fluid.vx,
        fluid.vy,
        fluid.dx,
        fluid.dt,
        fluid.solid_mask
    )

    advect_speed_numba_solid(
        fluid.vy,
        fluid.vy_tmp,
        fluid.vx,
        fluid.vy,
        fluid.dx,
        fluid.dt,
        fluid.solid_mask
    )

    fluid.vx, fluid.vx_tmp = fluid.vx_tmp, fluid.vx
    fluid.vy, fluid.vy_tmp = fluid.vy_tmp, fluid.vy

    # print("après advect")
    # mass_total(fluid.density, fluid.solid_mask, fluid.dx, fluid.m0)

    project_numba_solid(
        fluid,
        iterations
    )

    # abs_div = np.abs(fluid.div_tmp.copy())
    # abs_div[fluid.solid_mask] = 0

    # idx = np.unravel_index(np.argmax(abs_div), abs_div.shape)

    # print("deuxième proj")
    # print("max divergence:", abs_div[idx])
    # print("position:", idx)

    # print("après project")
    # mass_total(fluid.density, fluid.solid_mask, fluid.dx, fluid.m0)
    # print("\n")


def density_step(fluid: Fluid, iterations: int):

    # print("avant advect")
    # mass_total(fluid.density, fluid.solid_mask, fluid.dx, fluid.m0)

    advect_numba_solid(
        fluid.density,
        fluid.density_tmp,
        fluid.vx,
        fluid.vy,
        fluid.dx,
        fluid.dt,
        fluid.solid_mask
    )

    fluid.density, fluid.density_tmp = fluid.density_tmp, fluid.density

    # print("après advect")
    # mass_total(fluid.density, fluid.solid_mask, fluid.dx, fluid.m0)


    fluid.density, fluid.density_tmp = diffuse_smago_solid(
        fluid.density,
        fluid.density_tmp,
        fluid.diffuse_rhs_dens,
        fluid.density_diffusion,
        fluid.nu,
        fluid.normS,
        fluid.alpha,
        fluid.dt,
        fluid.dx,
        iterations,
        fluid.solid_mask
)

    # print("après diffuse")
    # mass_total(fluid.density, fluid.solid_mask, fluid.dx, fluid.m0)
    # print("\n")


def temperature_step(fluid: Fluid, iterations: int):

    advect_numba_solid(
        fluid.temperature,
        fluid.temperature_tmp,
        fluid.vx,
        fluid.vy,
        fluid.dx,
        fluid.dt,
        fluid.solid_mask
    )

    fluid.temperature, fluid.temperature_tmp = fluid.temperature_tmp, fluid.temperature

    fluid.temperature, fluid.temperature_tmp = diffuse_smago_solid(
        fluid.temperature,
        fluid.temperature_tmp,
        fluid.diffuse_rhs_temp,
        fluid.temperature_diffusion,
        fluid.nu,
        fluid.normS,
        fluid.alpha,
        fluid.dt,
        fluid.dx,
        iterations,
        fluid.solid_mask
    )


def step(fluid: Fluid, iterations: int = 150, solve_velocity: bool = True, solve_temperature: bool = False, solve_density: bool = False):

    if solve_velocity:
        velocity_step(fluid, iterations)

    if solve_density:
        density_step(fluid, iterations)

    if solve_temperature:
        temperature_step(fluid, iterations)