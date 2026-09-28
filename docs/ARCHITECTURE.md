# Architecture

```text
src/fluid2d/
├── state.py                    # simulation state + reusable buffers
├── diagnostics.py              # mass and simple diagnostics
├── boundaries/operators.py     # finite-difference operators
├── physics/
│   ├── forces.py
│   ├── smagorinsky.py
│   └── vorticity_confinement.py
├── solver/
│   ├── numba_mod.py             # generic advection / Poisson helpers
│   ├── solid.py                 # solid-aware operators
│   └── simulation.py            # timestep orchestration
└── render/
    └── renderer.py
```

The state object owns all long-lived NumPy arrays. Solver kernels mutate or swap preallocated buffers rather than creating full-size temporaries during a timestep.

The planned future architecture is backend-oriented:

```text
Scene/API
   ↓
Solver interface
   ├── cell-centered Stable Fluids
   ├── MAC incompressible solver
   ├── RANS
   └── compressible solver
```
