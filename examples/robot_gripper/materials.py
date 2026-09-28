"""Engineering material table for the gripper + cup, and per-part assignment.

Values are typical datasheet numbers (room temperature). ``strength`` is the
tensile yield (metals) or tensile strength (polymers / brittle materials), used
only to report a safety factor after simulation.
"""
import numpy as np

MATERIALS = {
    #  name                    E [Pa]    nu     rho [kg/m^3]  strength [Pa]
    "aluminium_6061_T6":  dict(E=68.9e9, nu=0.33, rho=2700.0, strength=276e6),
    "aluminium_7075_T6":  dict(E=71.7e9, nu=0.33, rho=2810.0, strength=503e6),
    "steel_440C":         dict(E=200e9,  nu=0.28, rho=7650.0, strength=1900e6),
    "silicone_shore_10A": dict(E=0.25e6, nu=0.47, rho=1070.0, strength=3.3e6),
    "polystyrene_GPPS":   dict(E=3.0e9,  nu=0.34, rho=1050.0, strength=40e6),
}

# part name -> material. Order of this dict is the winding-number priority:
# the *first* part whose generalized winding number exceeds 1/2 at a tet
# centroid wins (thin parts first, so bonded interfaces get the soft/thin part).
PART_MATERIAL = {
    "pad_left":       "silicone_shore_10A",
    "pad_right":      "silicone_shore_10A",
    "finger_left":    "aluminium_7075_T6",
    "finger_right":   "aluminium_7075_T6",
    "carriage_left":  "aluminium_6061_T6",
    "carriage_right": "aluminium_6061_T6",
    "rail":           "steel_440C",
    "housing":        "aluminium_6061_T6",
    "cup":            "polystyrene_GPPS",
}


def lame(E, nu):
    E, nu = np.asarray(E, float), np.asarray(nu, float)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam
