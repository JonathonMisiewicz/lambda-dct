from copy import deepcopy
import itertools

import numpy as np
from scipy import linalg as spla

from pilot_implementations import multilinear as mla
from pilot_implementations.multilinear.tensor import einsum

def simultaneous(en_nuc, h_ao, r_ao, start_orbitals,
        compute_intermediates, compute_energy, compute_orbital_residual,
        compute_amplitude_residual, compute_step, initialize_intermediates,
        niter=200, e_thresh=1e-13, r_thresh=1e-9, **kwargs):
    """
    Orbital-optimization algorithm with simultaneous optimization of orbitals and
    amplitudes.

    Input
    -----
    en_nuc: float
        Nuclear repulsion energy
    h_ao: np.ndarray
        AO basis one-electron integrals.
    r_ao: np.ndarray
        AO basis two-electron integrals
    compute_intermediates: function
    compute_energy: function
    compute_orbital_residual: function
    compute_amplitude_residual: function
    compute_step: function
    initialize_intermediates: function
        Method-specific functions. All take in a dict of intermediates. compute_energy returns an energy.
        All others modify the intermediate dict.
    niter: int
        The number of iterations before termination.
    e_thresh: float
        The energy convergence threshold.
    r_thresh: float
        The residual convergence threshold.

    Output
    ------
    intermediates: dict
        A dict of the various intermediates needed by the computation.
        Some of these can be big, so be careful!
    orbitals: dict
        A map from letters to a tuple of the alpha, beta np.ndarray
    """
    orbitals = deepcopy(start_orbitals)
    intermediates = initialize_intermediates(orbitals)
    prev_energy = en_nuc

    for iteration in range(niter):
        orbitals = mla.spinorb.orb_rot(intermediates, start_orbitals)

        # TODO: I may not need all of these integral blocks. And in fact, I probably don't.
        pairs = [''.join(x) for x in itertools.combinations_with_replacement("ov", 2)]

        h = mla.request_asym(h_ao, orbitals,
            pairs)
        g = mla.request_asym(r_ao, orbitals,
            [''.join(x) for x in itertools.combinations_with_replacement(pairs, 2)])
        for key, value in h.items():
            intermediates[f"h_{key}"] = value
        for key, value in g.items():
            intermediates[f"g_{key}"] = value

        compute_intermediates(intermediates)
        energy = compute_energy(intermediates) + en_nuc
        compute_orbital_residual(intermediates)
        compute_amplitude_residual(intermediates)
        compute_step(intermediates)

        deltaE = energy - prev_energy
        t1_norm = np.linalg.norm(intermediates["r1ov"])
        t2_norm = np.linalg.norm(intermediates["r2"])
        converged = np.fabs(deltaE) < e_thresh and t1_norm < r_thresh and t2_norm < r_thresh
        print(f"{iteration:3d} {energy:20.14f} {t1_norm:20.14f} {t2_norm:20.14f}", flush=True)
        prev_energy = energy
        if converged: break
    if not converged:
        print("CONVERGENCE FAIL")
        raise Exception
    else:
        intermediates["energy"] = energy
        print("CONVERGENCE SUCCESS")
   

    print("Partial Trace Sanity Check")
    partial_trace = np.trace(intermediates["rdm_oo"]) + np.trace(intermediates["rdm_vv"])
    partial_traced_oo = np.trace(intermediates["rdm_oooo"], axis1=1, axis2=3) + np.trace(intermediates["rdm_ovov"], axis1=1, axis2=3)
    print(spla.norm(partial_traced_oo - intermediates["rdm_oo"] * (partial_trace - 1)))
    partial_traced_vv = np.trace(intermediates["rdm_vvvv"], axis1=0, axis2=2) + np.trace(intermediates["rdm_ovov"], axis1=0, axis2=2)
    print(spla.norm(partial_traced_vv - intermediates["rdm_vv"] * (partial_trace - 1)))

    return intermediates, orbitals

