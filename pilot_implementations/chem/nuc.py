import itertools
from typing import Iterable

import numpy as np

from pilot_implementations.chem import elements

def energy(geom: Iterable[tuple[str, tuple[float, float, float]]]) -> float:
    """
    Compute the nuclear repulsion energy.

    geom:
        Each entry is the atomic label, then a tuple of the Cartesian coordinates in bohr.
    """
    energy = 0
    for combination in itertools.combinations(geom, 2):
        labels, coords = zip(*combination)
        charges = tuple(map(elements.charge, labels))
        del_r = np.linalg.norm(np.subtract(*coords))
        energy += charges[0] * charges[1] / del_r
    return energy
