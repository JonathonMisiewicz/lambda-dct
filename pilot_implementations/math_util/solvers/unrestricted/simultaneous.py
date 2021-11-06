from copy import deepcopy

import numpy as np

from pilot_implementations import multilinear as mla
from . import proc

def simultaneous(en_nuc, h_ao, r_ao, start_orbitals,
        compute_intermediates, compute_energy, compute_orbital_residual,
        compute_amplitude_residual, compute_step, initialize_intermediates,
        niter=200, e_thresh=1e-13, r_thresh=1e-9, **kwargs):
    orbitals = deepcopy(start_orbitals)
    intermediates = initialize_intermediates(orbitals)
    prev_energy = en_nuc

    for iteration in range(niter):
        orbitals = mla.spinorb.orb_rot_SI(intermediates, start_orbitals)

        proc.compute_integrals(intermediates, h_ao, r_ao, orbitals)

        compute_intermediates(intermediates)
        energy = compute_energy(intermediates) + en_nuc
        compute_orbital_residual(intermediates)
        compute_amplitude_residual(intermediates)
        compute_step(intermediates)

        deltaE = energy - prev_energy
        t1_norm = np.linalg.norm([np.linalg.norm(intermediates["r1_ov_α"]), np.linalg.norm(intermediates["r1_ov_β"])])
        t2_norm = np.linalg.norm([np.linalg.norm(intermediates["r2_αα"]), np.linalg.norm(intermediates["r2_αβ"]), np.linalg.norm(intermediates["r2_ββ"])])
        converged = np.fabs(deltaE) < e_thresh and t1_norm < r_thresh and t2_norm < r_thresh
        print_str = f"{iteration:3d} {energy:20.14f} {t1_norm:20.14f} {t2_norm:20.14f}"

        if "r3" in intermediates:
            t3_norm = np.linalg.norm(intermediates["r3_ααα"])
            converged = converged and t3_norm < r_thresh
            print_str += f" {t3_norm:20.14f}"
        elif "r3_ααα" in intermediates:
            t3_norm = np.linalg.norm([
                    np.linalg.norm(intermediates["r3_ααα"]),
                    np.linalg.norm(intermediates["r3_ααβ"]),
                    np.linalg.norm(intermediates["r3_αββ"]),
                    np.linalg.norm(intermediates["r3_βββ"])
                ])
            converged = converged and t3_norm < r_thresh
            print_str += f" {t3_norm:20.14f}"

        print(print_str, flush=True)
        prev_energy = energy
        if converged: break
    if not converged:
        print("CONVERGENCE FAIL")
        raise Exception
    else:
        intermediates["energy"] = energy
        print("CONVERGENCE SUCCESS")

    return intermediates, orbitals

