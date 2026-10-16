"""Default physical values for HydroMorph-Analytics.

Nothing here is a hidden literal inside a solver. Every entry is surfaced in the
UI as an editable, unit-labelled input and handed to the engines explicitly, so
the user (not the code) owns g, rho, nu and kappa.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Constant:
    key: str
    label: str
    default: float
    min_value: float
    max_value: float
    fmt: str = "%.4g"
    help: str = ""


GRAVITY = Constant("g", "Gravitational Acceleration g (m/s²)", 9.81, 0.1, 30.0,
                   help="Default is standard Earth gravity.")
WATER_DENSITY = Constant("rho_w", "Water Density ρ_w (kg/m³)", 1000.0, 500.0, 2000.0)
SEDIMENT_DENSITY = Constant("rho_s", "Sediment Grain Density ρ_s (kg/m³)", 2650.0, 500.0, 10000.0,
                            help="Default is quartz sand.")
KINEMATIC_VISCOSITY = Constant("nu", "Kinematic Viscosity ν (m²/s)", 1.0e-6, 1.0e-8, 1.0e-2, "%.3e",
                               help="Default is clear water near 20 °C.")
VON_KARMAN = Constant("kappa", "von Kármán Constant κ (dimensionless)", 0.41, 0.2, 0.6)
