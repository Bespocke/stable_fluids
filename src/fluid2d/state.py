import numpy as np


class Fluid:
    def __init__(self, N: int, dx: float, dt: float):

        # ============================================================
        # Simulation parameters
        # ============================================================

        self.N = N
        self.dx = dx
        self.dt = dt

        # ============================================================
        # Velocity field
        # ============================================================

        self.vx = np.zeros((N, N), dtype=np.float32)
        self.vy = np.zeros((N, N), dtype=np.float32)

        # Buffers
        self.vx_tmp = np.zeros((N, N), dtype=np.float32)
        self.vy_tmp = np.zeros((N, N), dtype=np.float32)

        # ============================================================
        # Pressure projection
        # ============================================================

        self.pressure = np.zeros((N, N), dtype=np.float32)
        self.pressure_tmp = np.zeros((N, N), dtype=np.float32)

        # Divergence (right-hand side of Poisson equation)
        self.div_tmp = np.zeros((N, N), dtype=np.float32)

        # ============================================================
        # Passive scalar fields
        # ============================================================

        # Density
        self.density = np.zeros((N, N), dtype=np.float32)
        self.density_tmp = np.zeros((N, N), dtype=np.float32)

        # Temperature
        self.temperature = np.zeros((N, N), dtype=np.float32)
        self.temperature_tmp = np.zeros((N, N), dtype=np.float32)

        # ============================================================
        # Diffusion solver buffers
        # RHS = Right-Hand Side of the linear system
        # ============================================================

        self.diffuse_rhs_vx = np.zeros((N, N), dtype=np.float32)
        self.diffuse_rhs_vy = np.zeros((N, N), dtype=np.float32)

        self.diffuse_rhs_dens = np.zeros((N, N), dtype=np.float32)
        self.diffuse_rhs_temp = np.zeros((N, N), dtype=np.float32)

        # ============================================================
        # Vorticity confinement
        # ============================================================

        # Curl magnitude
        self.curl = np.zeros((N, N), dtype=np.float32)
        self.curl_abs = np.zeros((N, N), dtype=np.float32)

        # Normalized curl gradient
        self.Nx = np.zeros((N, N), dtype=np.float32)
        self.Ny = np.zeros((N, N), dtype=np.float32)

        # ============================================================
        # Temporary operator buffers
        # Reused by grad(), divergence(), normalize(), etc.
        # ============================================================

        self.gradx_tmp = np.zeros((N, N), dtype=np.float32)
        self.grady_tmp = np.zeros((N, N), dtype=np.float32)
        self.gradx_tmp2 = np.zeros((N, N), dtype=np.float32)
        self.grady_tmp2 = np.zeros((N, N), dtype=np.float32)
        self.norm_tmp = np.zeros((N, N), dtype=np.float32)

        # ============================================================
        # Geometry
        # ============================================================

        # 0 = fluid
        # 1 = solid

        # ============================================================
        # Physical coefficients
        # ============================================================

        self.density_diffusion = 0.0
        self.temperature_diffusion = 0.0

        # ============================================================
        # Smagorinsky
        # ============================================================
        
        self.S = np.zeros((N, N, 3), dtype=np.float32) # diffuse 1
        self.normS = np.zeros((N, N), dtype=np.float32) # diffuse 1
        self.nu = np.zeros((N, N), dtype=np.float32) # tous
        self.alpha = np.zeros((N, N), dtype=np.float32) # diffuse 1
        self.viscosity = 0.0

        self.nuE = np.zeros((N, N), dtype=np.float32)
        self.nuW = np.zeros((N, N), dtype=np.float32)
        self.nuN = np.zeros((N, N), dtype=np.float32)
        self.nuS = np.zeros((N, N), dtype=np.float32)
        self.a = np.zeros((N, N), dtype=np.float32)
        self.inv = np.zeros((N, N), dtype=np.float32)

        # ============================================================
        # Obstacles
        # ============================================================

        self.solid_mask = np.zeros((N, N), dtype=np.bool_)

        # ============================================================
        # Masse
        # ============================================================

        self.m0 = 0.0

        # ============================================================
        # Forces
        # ============================================================

        self.forces = np.zeros((2,), dtype=np.float32)
