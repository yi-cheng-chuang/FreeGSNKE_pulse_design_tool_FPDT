# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:34:51 2026

@author: bookb
"""

"""Minimal FreeGSNKE static inverse + forward example."""
import numpy as np
import matplotlib.pyplot as plt

from freegsnke import build_machine, equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp
from freegsnke.inverse import Inverse_optimizer


def ellipse(R0, N, R_minor, kappa=1.0):
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    return R0 + R_minor * np.cos(theta), R_minor * kappa * np.sin(theta)


# --- toy machine (Anamak-style) ---
R0 = 1.0
r_c, z_c = ellipse(R0, N=8, R_minor=0.70)
active = {
    i: {
        "R": [r], "Z": [z], "dR": 0.05, "dZ": 0.05,
        "resistivity": 1.55e-8, "polarity": 1, "multiplier": 1,
    }
    for i, (r, z) in enumerate(zip(r_c, z_c))
}

r_p, z_p = ellipse(R0, N=40, R_minor=0.50)
passive = [
    {"R": r, "Z": z, "dR": 0.02, "dZ": 0.02, "resistivity": 5.5e-7}
    for r, z in zip(r_p, z_p)
]

r_l, z_l = ellipse(R0, N=101, R_minor=0.45)
limiter = [{"R": r, "Z": z} for r, z in zip(r_l, z_l)]

tokamak = build_machine.tokamak(
    active_coils_data=active,
    passive_coils_data=passive,
    limiter_data=limiter,
    wall_data=limiter,
)

# Grid nx, ny must be 2**n + 1
eq = equilibrium_update.Equilibrium(
    tokamak=tokamak,
    Rmin=0.1, Rmax=1.9, Zmin=-0.8, Zmax=0.8,
    nx=65, ny=129,
)

profiles = ConstrainPaxisIp(
    eq=eq, paxis=8e3, Ip=2e5, fvac=0.5 * R0,
    alpha_m=1.8, alpha_n=1.2,
)

solver = GSstaticsolver.NKGSsolver(eq)

# Inverse: find coil currents for a target shape
Rx, Zx, Rout = 0.80, 0.35, 1.22
constrain = Inverse_optimizer(
    null_points=[[Rx, Rx], [Zx, -Zx]],
    isoflux_set=np.array([[[Rx, Rx, Rout], [Zx, -Zx, 0.0]]]),
    weight_isoflux=1.0,
    weight_nulls=0.8,
)

solver.inverse_solve(
    eq=eq, profiles=profiles, constrain=constrain,
    target_relative_tolerance=1e-6, verbose=True,
)
print("Coil currents after inverse:", eq.tokamak.getCurrents())

# Forward: hold those currents, re-solve GS
solver.solve(
    eq=eq, profiles=profiles, constrain=None,
    target_relative_tolerance=1e-8, verbose=True,
)

print("Ip [A] =", eq.plasmaCurrent())
print("magnetic axis (R, Z) =", eq.Rmagnetic(), eq.Zmagnetic())

fig, ax = plt.subplots(figsize=(5, 6))
eq.plot(axis=ax, show=False)
tokamak.plot(axis=ax, show=False)
ax.set_aspect("equal")
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
plt.tight_layout()
plt.savefig("freegsnke_example.png", dpi=150)
plt.show()