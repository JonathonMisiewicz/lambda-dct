from copy import deepcopy
import itertools

import numpy as np

from pilot_implementations import multilinear as mla

def vanilla(en_nuc, h_ao, r_ao, start_orbitals,
        compute_intermediates, compute_energy,
        compute_amplitude_residual, compute_step, initialize_intermediates,
        niter=200, e_thresh=1e-13, r_thresh=1e-9,
        check_minima = False,
        **kwargs):
    """
    Straightforward non-OO algorithm.

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

    for iteration in range(niter):
        compute_intermediates(intermediates)
        energy = compute_energy(intermediates) + en_nuc
        compute_amplitude_residual(intermediates)
        compute_step(intermediates)

        deltaE = energy - prev_energy
        t1_norm = np.linalg.norm(intermediates.get("r1", 0))
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

    # Numerical differentiate.
    if check_minima:
        h = 0.00005
        tensor = intermediates["t1"]
        with np.nditer(tensor, flags=["multi_index"], op_flags=["readwrite"]) as it:
            for x in it:
                tensor[it.multi_index] += h
                compute_intermediates(intermediates)
                ep = compute_energy(intermediates) + en_nuc
                tensor[it.multi_index] -= 2 * h
                compute_intermediates(intermediates)
                em = compute_energy(intermediates) + en_nuc
                tensor[it.multi_index] += h
                try:
                    assert np.abs((ep - em) / (2 * h)) < 5e-9
                except AssertionError:
                    print(np.abs((ep - em) / (2 * h)))
                    raise AssertionError

        tensor = intermediates["t2"]
        with np.nditer(tensor, flags=["multi_index"], op_flags=["readwrite"]) as it:
            for x in it:
                if np.abs(tensor[it.multi_index]) < 1e-10:
                    continue
                i1 = it.multi_index
                i2 = i1[0], i1[1], i1[3], i1[2]
                i3 = i1[1], i1[0], i1[2], i1[3]
                i4 = i1[1], i1[0], i1[3], i1[2]
                tensor[i1] += h
                tensor[i4] += h
                tensor[i2] -= h
                tensor[i3] -= h
                compute_intermediates(intermediates)
                ep = compute_energy(intermediates)
                tensor[i1] -= 2 * h
                tensor[i4] -= 2 * h
                tensor[i2] += 2 * h
                tensor[i3] += 2 * h
                compute_intermediates(intermediates)
                em = compute_energy(intermediates)
                tensor[i1] += h
                tensor[i4] += h
                tensor[i2] -= h
                tensor[i3] -= h
                try:
                    assert np.abs((ep - em) / (2 * h)) < 5e-9
                except AssertionError:
                    print(np.abs((ep - em) / (2 * h)))
                    raise Exception

    return intermediates, orbitals

