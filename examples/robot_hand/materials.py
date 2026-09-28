"""Materials for the Allegro hand + cup, and part -> material assignment.

The Allegro Hand's palm and phalanges are machined aluminium housings around
the joint motors; the joint output shafts are hardened steel; the white
fingertips are moulded polyurethane rubber over an aluminium mount. The cup is
the same brittle polystyrene cup as in the gripper example.
"""
import numpy as np

MATERIALS = {
    #  name                    E [Pa]    nu     rho [kg/m^3]  strength [Pa]
    "aluminium_6061_T6":  dict(E=68.9e9, nu=0.33, rho=2700.0, strength=276e6),
    "aluminium_7075_T6":  dict(E=71.7e9, nu=0.33, rho=2810.0, strength=503e6),
    "steel_AISI_4140":    dict(E=205e9,  nu=0.29, rho=7850.0, strength=655e6),
    "polyurethane_40A":   dict(E=1.5e6,  nu=0.48, rho=1100.0, strength=10e6),
    "polystyrene_GPPS":   dict(E=3.0e9,  nu=0.34, rho=1050.0, strength=40e6),
}


def part_material(part: str) -> str:
    if part == "cup":
        return "polystyrene_GPPS"
    if part.endswith("_shaft"):
        return "steel_AISI_4140"
    if part.endswith("_tip_rubber"):
        return "polyurethane_40A"
    if part == "palm" or part.endswith("_tip_core"):
        return "aluminium_6061_T6"
    return "aluminium_7075_T6"


def part_priority(part: str) -> int:
    """Winding-number priority (lower wins): thin/embedded parts first."""
    for i, key in enumerate(["_shaft", "_tip_rubber", "_tip_core"]):
        if part.endswith(key):
            return i
    return 3


def lame(E, nu):
    E, nu = np.asarray(E, float), np.asarray(nu, float)
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
