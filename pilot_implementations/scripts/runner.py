from ..qc_codes import psi as program
import itertools
import numpy as np
from .. import math_util, chem
from scipy import linalg as spla
from .. import multilinear as mla
from pilot_implementations.multilinear.tensor import einsum
from pilot_implementations.math_util.solvers.spinorbital import proc as spinorbital_proc

def subspace(molecule, solver, test=False, comp_grad=False, e_thresh=1e-14, r_thresh=1e-9, **kwargs):
    CHARGE = molecule["charge"]
    NUM_UNPAIRED = molecule["num_unpaired"]
    BASIS = molecule["basis"]
    GEOM = molecule["geom"]

    atoms = [i[0] for i in GEOM]

    # The following integrals are in the atomic orbital basis.
    h_ao = program.core_hamiltonian(BASIS, GEOM)
    r_ao = program.repulsion(BASIS, GEOM)

    h_aso = mla.to_spinorb(h_ao, electron_indices=((0, 1),))
    r_aso = mla.to_spinorb(r_ao, electron_indices=((0, 2), (1, 3)))
    g_aso = r_aso - np.transpose(r_aso, (0, 1, 3, 2))

    try:
        orbitals = program.read_orbitals(BASIS, GEOM)
    except AssertionError:
        orbitals = program.unrestricted_orbitals(BASIS, GEOM, CHARGE, NUM_UNPAIRED, **kwargs)
    en_nuc = chem.nuc.energy(GEOM)

    intermed, orbitals = solver(en_nuc, h_ao, r_ao, orbitals, e_thresh=e_thresh, r_thresh=r_thresh, check_minima=kwargs.get("check_minima", False))

    if comp_grad and not hasattr(solver, "SI"):
        nx = program.nuclear_potential_deriv(GEOM)
        oei = spinorbital_proc.backtransform_hermitian_opdm(orbitals, intermed)
        tei = spinorbital_proc.backtransform_hermitian_tpdm(orbitals, intermed)
        gei = spinorbital_proc.backtransform_hermitian_gfm(orbitals, intermed)
        grad = np.zeros(nx.shape)
        for atom in range(len(GEOM)):
            hx_ao = program.core_hamiltonian_grad(BASIS, GEOM, atom)
            rx_ao = program.repulsion_grad(BASIS, GEOM, atom)
            sx_ao = program.overlap_grad(BASIS, GEOM, atom)
            for i, (h, r, s) in enumerate(zip(hx_ao, rx_ao, sx_ao)):
                hx_aso = mla.to_spinorb(h, electron_indices=((0,1), ))
                sx_aso = mla.to_spinorb(s, electron_indices=((0,1), ))
                rx_aso = mla.to_spinorb(r, electron_indices=((0,2), (1,3)))
                gx_aso = rx_aso - np.transpose(rx_aso, (0, 1, 3, 2))
                grad[atom][i] = math_util.solvers.common.perturbation_gradient(gei, hx_aso, gx_aso, sx_aso, nx[atom][i], oei, tei)
        print(grad)
        intermed["gradient"] = grad
    elif comp_grad and hasattr(solver, "SI"):
        nx = program.nuclear_potential_deriv(GEOM)
        grad = np.zeros(nx.shape)
        opdm = einsum("pq, Pp, Qq -> PQ", intermed["rdm_oo_α"], orbitals["o"][0], orbitals["o"][0]) + (
            einsum("pq, Pp, Qq -> PQ", intermed["rdm_oo_β"], orbitals["o"][1], orbitals["o"][1])) + (
            einsum("pq, Pp, Qq -> PQ", intermed["rdm_vv_α"], orbitals["v"][0], orbitals["v"][0])) + (
            einsum("pq, Pp, Qq -> PQ", intermed["rdm_vv_β"], orbitals["v"][1], orbitals["v"][1]))
        tpdm = 2 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oooo_αα"], orbitals["o"][0], orbitals["o"][0], orbitals["o"][0], orbitals["o"][0]) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oooo_αβ"], orbitals["o"][0], orbitals["o"][1], orbitals["o"][0], orbitals["o"][1])) + (
            2 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oooo_ββ"], orbitals["o"][1], orbitals["o"][1], orbitals["o"][1], orbitals["o"][1])) + (
            2 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vvvv_αα"], orbitals["v"][0], orbitals["v"][0], orbitals["v"][0], orbitals["v"][0])) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vvvv_αβ"], orbitals["v"][0], orbitals["v"][1], orbitals["v"][0], orbitals["v"][1])) + (
            2 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vvvv_ββ"], orbitals["v"][1], orbitals["v"][1], orbitals["v"][1], orbitals["v"][1])) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oovv_αα"], orbitals["o"][0], orbitals["o"][0], orbitals["v"][0], orbitals["v"][0])) + (
            8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oovv_αβ"], orbitals["o"][0], orbitals["o"][1], orbitals["v"][0], orbitals["v"][1])) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oovv_ββ"], orbitals["o"][1], orbitals["o"][1], orbitals["v"][1], orbitals["v"][1])) + (
            4 * mla.antisymmetrize_axes_plus(
                einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovov_αα"], orbitals["o"][0], orbitals["v"][0], orbitals["o"][0], orbitals["v"][0]),
                 ((0,), (1,)) )) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovov_αβ"], orbitals["o"][0], orbitals["v"][1], orbitals["o"][0], orbitals["v"][1])) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vovo_αβ"], orbitals["v"][0], orbitals["o"][1], orbitals["v"][0], orbitals["o"][1])) + (
            8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovvo_αβ"], orbitals["o"][0], orbitals["v"][1], orbitals["v"][0], orbitals["o"][1])) + (
            4 * mla.antisymmetrize_axes_plus(
                einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovov_ββ"], orbitals["o"][1], orbitals["v"][1], orbitals["o"][1], orbitals["v"][1]),
                 ((0,), (1,)) ))
                
        gen_fock_oo_α = einsum("i J, I i -> IJ", intermed["h_oo_α"], intermed["rdm_oo_α"]) + (
            0.5 * einsum("jk Ji, Ii jk -> IJ", intermed["g_oooo_αα"], intermed["rdm_oooo_αα"])) + (
            einsum("jk Ji, Ii jk -> IJ", intermed["g_oooo_αβ"], intermed["rdm_oooo_αβ"])) + (
            0.5 * einsum("Ji ab, Ii ab -> IJ", intermed["g_oovv_αα"], intermed["rdm_oovv_αα"])) + (
            einsum("Ji ab, Ii ab -> IJ", intermed["g_oovv_αβ"], intermed["rdm_oovv_αβ"])) + (
            einsum("ib Ja, Ia ib -> IJ", intermed["g_ovov_αα"], intermed["rdm_ovov_αα"])) + (
            einsum("ib Ja, Ia ib -> IJ", intermed["g_ovov_αβ"], intermed["rdm_ovov_αβ"])) + (
            einsum("Ja bi, Ia bi -> IJ", intermed["g_ovvo_αβ"], intermed["rdm_ovvo_αβ"]))
        gen_fock_oo_β = einsum("i J, I i -> IJ", intermed["h_oo_β"], intermed["rdm_oo_β"]) + (
            0.5 * einsum("jk Ji, Ii jk -> IJ", intermed["g_oooo_ββ"], intermed["rdm_oooo_ββ"])) + (
            einsum("jk iJ, iI jk -> IJ", intermed["g_oooo_αβ"], intermed["rdm_oooo_αβ"])) + (
            0.5 * einsum("Ji ab, Ii ab -> IJ", intermed["g_oovv_ββ"], intermed["rdm_oovv_ββ"])) + (
            einsum("iJ ab, iI ab -> IJ", intermed["g_oovv_αβ"], intermed["rdm_oovv_αβ"])) + (
            einsum("ib Ja, Ia ib -> IJ", intermed["g_ovov_ββ"], intermed["rdm_ovov_ββ"])) + (
            einsum("bi aJ, aI bi -> IJ", intermed["g_vovo_αβ"], intermed["rdm_vovo_αβ"])) + (
            einsum("ib aJ, ib aI -> IJ", intermed["g_ovvo_αβ"], intermed["rdm_ovvo_αβ"]))
        gen_fock_vv_α = einsum("a B, A a -> AB", intermed["h_vv_α"], intermed["rdm_vv_α"]) + (
            0.5 * einsum("bc Ba, Aa bc -> AB", intermed["g_vvvv_αα"], intermed["rdm_vvvv_αα"])) + (
            einsum("bc Ba, Aa bc -> AB", intermed["g_vvvv_αβ"], intermed["rdm_vvvv_αβ"])) + (
            0.5 * einsum("ij Ba, ij Aa -> AB", intermed["g_oovv_αα"], intermed["rdm_oovv_αα"])) + (
            einsum("ij Ba, ij Aa -> AB", intermed["g_oovv_αβ"], intermed["rdm_oovv_αβ"])) + (
            einsum("iB ja, iA ja -> AB", intermed["g_ovov_αα"], intermed["rdm_ovov_αα"])) + (
            einsum("Bi aj, Ai aj -> AB", intermed["g_vovo_αβ"], intermed["rdm_vovo_αβ"])) + (
            einsum("ja Bi, ja Ai -> AB", intermed["g_ovvo_αβ"], intermed["rdm_ovvo_αβ"]))
        gen_fock_vv_β = einsum("a B, A a -> AB", intermed["h_vv_β"], intermed["rdm_vv_β"]) + (
            0.5 * einsum("bc Ba, Aa bc -> AB", intermed["g_vvvv_ββ"], intermed["rdm_vvvv_ββ"])) + (
            einsum("bc aB, aA bc -> AB", intermed["g_vvvv_αβ"], intermed["rdm_vvvv_αβ"])) + (
            0.5 * einsum("ij Ba, ij Aa -> AB", intermed["g_oovv_ββ"], intermed["rdm_oovv_ββ"])) + (
            einsum("ij aB, ij aA -> AB", intermed["g_oovv_αβ"], intermed["rdm_oovv_αβ"])) + (
            einsum("iB ja, iA ja -> AB", intermed["g_ovov_ββ"], intermed["rdm_ovov_ββ"])) + (
            einsum("iB ja, iA ja -> AB", intermed["g_ovov_αβ"], intermed["rdm_ovov_αβ"])) + (
            einsum("iB aj, iA aj -> AB", intermed["g_ovvo_αβ"], intermed["rdm_ovvo_αβ"]))
        gen_fock_ov_α = einsum("Ia, Ii -> ia", intermed["h_ov_α"], intermed["rdm_oo_α"]) + (
            -0.5 * einsum("JK Ia, iI JK -> ia", intermed["g_ooov_αα"], intermed["rdm_oooo_αα"])) + (
            einsum("JK aI, iI JK -> ia", intermed["g_oovo_αβ"], intermed["rdm_oooo_αβ"])) + (
            -0.5 * einsum("Ia AB, iI AB -> ia", intermed["g_ovvv_αα"], intermed["rdm_oovv_αα"])) + (
            einsum("aI AB, iI AB -> ia", intermed["g_vovv_αβ"], intermed["rdm_oovv_αβ"])) + (
            einsum("IB aA, iA IB -> ia", intermed["g_ovvv_αα"], intermed["rdm_ovov_αα"])) + (
            einsum("IB aA, iA IB -> ia", intermed["g_ovvv_αβ"], intermed["rdm_ovov_αβ"])) + (
            einsum("BI aA, iA BI -> ia", intermed["g_vovv_αβ"], intermed["rdm_ovvo_αβ"]))
        gen_fock_ov_β = einsum("Ia, Ii -> ia", intermed["h_ov_β"], intermed["rdm_oo_β"]) + (
            -0.5 * einsum("JK Ia, iI JK -> ia", intermed["g_ooov_ββ"], intermed["rdm_oooo_ββ"])) + (
            einsum("JK Ia, Ii JK -> ia", intermed["g_ooov_αβ"], intermed["rdm_oooo_αβ"])) + (
            -0.5 * einsum("Ia AB, iI AB -> ia", intermed["g_ovvv_ββ"], intermed["rdm_oovv_ββ"])) + (
            einsum("Ia AB, Ii AB -> ia", intermed["g_ovvv_αβ"], intermed["rdm_oovv_αβ"])) + (
            einsum("IB aA, iA IB -> ia", intermed["g_ovvv_ββ"], intermed["rdm_ovov_ββ"])) + (
            einsum("BI Aa, Ai BI -> ia", intermed["g_vovv_αβ"], intermed["rdm_vovo_αβ"])) + (
            einsum("IB Aa, IB Ai -> ia", intermed["g_ovvv_αβ"], intermed["rdm_ovvo_αβ"]))
        gen_fock_vo_α = einsum("iA, Aa -> ai", intermed["h_ov_α"], intermed["rdm_vv_α"]) + (
            0.5 * einsum("iA BC, aA BC -> ai", intermed["g_ovvv_αα"], intermed["rdm_vvvv_αα"])) + (
            einsum("iA BC, aA BC -> ai", intermed["g_ovvv_αβ"], intermed["rdm_vvvv_αβ"])) + (
            0.5 * einsum("IJ iA, IJ aA -> ai", intermed["g_ooov_αα"], intermed["rdm_oovv_αα"])) + (
            einsum("IJ iA, IJ aA -> ai", intermed["g_ooov_αβ"], intermed["rdm_oovv_αβ"])) + (
            einsum("Ii JA, Ia JA -> ai", intermed["g_ooov_αα"], intermed["rdm_ovov_αα"])) + (
            einsum("iI JA, JA aI -> ai", intermed["g_ooov_αβ"], intermed["rdm_ovvo_αβ"])) + (
            einsum("iI AJ, aI AJ -> ai", intermed["g_oovo_αβ"], intermed["rdm_vovo_αβ"]))
        gen_fock_vo_β = einsum("iA, Aa -> ai", intermed["h_ov_β"], intermed["rdm_vv_β"]) + (
            0.5 * einsum("iA BC, aA BC -> ai", intermed["g_ovvv_ββ"], intermed["rdm_vvvv_ββ"])) + (
            einsum("Ai CB, Aa CB -> ai", intermed["g_vovv_αβ"], intermed["rdm_vvvv_αβ"])) + (
            0.5 * einsum("IJ iA, IJ aA -> ai", intermed["g_ooov_ββ"], intermed["rdm_oovv_ββ"])) + (
            einsum("JI Ai, JI Aa -> ai", intermed["g_oovo_αβ"], intermed["rdm_oovv_αβ"])) + (
            einsum("Ii JA, Ia JA -> ai", intermed["g_ooov_ββ"], intermed["rdm_ovov_ββ"])) + (
            einsum("Ii JA, Ia JA -> ai", intermed["g_ooov_αβ"], intermed["rdm_ovov_αβ"])) + (
            einsum("Ii AJ, Ia AJ -> ai", intermed["g_oovo_αβ"], intermed["rdm_ovvo_αβ"]))
        if "rdm_ov_α" in intermed:
            opdm += 2 * einsum("pq, Pp, Qq -> PQ", intermed["rdm_ov_α"], orbitals["o"][0], orbitals["v"][0])
            opdm += 2 * einsum("pq, Pp, Qq -> PQ", intermed["rdm_ov_β"], orbitals["o"][1], orbitals["v"][1])
            gen_fock_oo_α += einsum("J a, I a -> IJ", intermed["h_ov_α"], intermed["rdm_ov_α"])
            gen_fock_oo_β += einsum("J a, I a -> IJ", intermed["h_ov_β"], intermed["rdm_ov_β"])
            gen_fock_ov_α += einsum("A a, I a -> IA", intermed["h_vv_α"], intermed["rdm_ov_α"])
            gen_fock_ov_β += einsum("A a, I a -> IA", intermed["h_vv_β"], intermed["rdm_ov_β"])
            gen_fock_vo_α += einsum("I i, i A -> AI", intermed["h_oo_α"], intermed["rdm_ov_α"])
            gen_fock_vo_β += einsum("I i, i A -> AI", intermed["h_oo_β"], intermed["rdm_ov_β"])
            gen_fock_vv_α += einsum("i B, i A -> AB", intermed["h_ov_α"], intermed["rdm_ov_α"])
            gen_fock_vv_β += einsum("i B, i A -> AB", intermed["h_ov_β"], intermed["rdm_ov_β"])
        if "rdm_ooov_αα" in intermed:
            gen_fock_ov_α +=  einsum("IaJA, IiJA-> ia", intermed["g_ovov_αα"], intermed["rdm_ooov_αα"])
            gen_fock_ov_α +=  einsum("JAaI, iIJA-> ia", intermed["g_ovvo_αβ"], intermed["rdm_ooov_αβ"])
            gen_fock_ov_α +=  einsum("aIAJ, iIAJ-> ia", intermed["g_vovo_αβ"], intermed["rdm_oovo_αβ"])
            gen_fock_ov_β +=  einsum("IaJA, IiJA-> ia", intermed["g_ovov_ββ"], intermed["rdm_ooov_ββ"])
            gen_fock_ov_β +=  einsum("IaJA, IiJA-> ia", intermed["g_ovov_αβ"], intermed["rdm_ooov_αβ"])
            gen_fock_ov_β +=  einsum("IaAJ, IiAJ-> ia", intermed["g_ovvo_αβ"], intermed["rdm_oovo_αβ"])
            gen_fock_ov_α += 0.5 * einsum("IJaA, IJiA -> ia", intermed["g_oovv_αα"], intermed["rdm_ooov_αα"])
            gen_fock_ov_α += einsum("IJaA, IJiA -> ia", intermed["g_oovv_αβ"], intermed["rdm_ooov_αβ"])
            gen_fock_ov_β += 0.5 * einsum("IJaA, IJiA -> ia", intermed["g_oovv_ββ"], intermed["rdm_ooov_ββ"])
            gen_fock_ov_β += einsum("IJAa, IJAi -> ia", intermed["g_oovv_αβ"], intermed["rdm_oovo_αβ"])
            gen_fock_ov_α += 0.5 * einsum("aABC, iABC-> ia", intermed["g_vvvv_αα"], intermed["rdm_ovvv_αα"])
            gen_fock_ov_α += einsum("aABC, iABC-> ia", intermed["g_vvvv_αβ"], intermed["rdm_ovvv_αβ"])
            gen_fock_ov_β += 0.5 * einsum("aABC, iABC-> ia", intermed["g_vvvv_ββ"], intermed["rdm_ovvv_ββ"])
            gen_fock_ov_β += einsum("AaBC, AiBC-> ia", intermed["g_vvvv_αβ"], intermed["rdm_vovv_αβ"])

            gen_fock_vo_α += 0.5 * einsum("IiAB, IaAB -> ai", intermed["g_oovv_αα"], intermed["rdm_ovvv_αα"])
            gen_fock_vo_α += einsum("iIAB, aIAB -> ai", intermed["g_oovv_αβ"], intermed["rdm_vovv_αβ"])
            gen_fock_vo_β += 0.5 * einsum("IiAB, IaAB -> ai", intermed["g_oovv_ββ"], intermed["rdm_ovvv_ββ"])
            gen_fock_vo_β += einsum("IiAB, IaAB -> ai", intermed["g_oovv_αβ"], intermed["rdm_ovvv_αβ"])
            gen_fock_vo_α += einsum("IAiB, IAaB -> ai", intermed["g_ovov_αα"], intermed["rdm_ovvv_αα"])
            gen_fock_vo_α += einsum("IAiB, IAaB -> ai", intermed["g_ovov_αβ"], intermed["rdm_ovvv_αβ"])
            gen_fock_vo_α += einsum("iBAI, AIaB -> ai", intermed["g_ovvo_αβ"], intermed["rdm_vovv_αβ"])
            gen_fock_vo_β += einsum("IAiB, IAaB -> ai", intermed["g_ovov_ββ"], intermed["rdm_ovvv_ββ"])
            gen_fock_vo_β += einsum("IABi, IABa -> ai", intermed["g_ovvo_αβ"], intermed["rdm_ovvv_αβ"])
            gen_fock_vo_β += einsum("AIBi, AIBa -> ai", intermed["g_vovo_αβ"], intermed["rdm_vovv_αβ"])
            gen_fock_vo_α += 0.5 * einsum("IJKi, IJKa -> ai", intermed["g_oooo_αα"], intermed["rdm_ooov_αα"])
            gen_fock_vo_α += einsum("IJiK, IJaK -> ai", intermed["g_oooo_αβ"], intermed["rdm_oovo_αβ"])
            gen_fock_vo_β += 0.5 * einsum("IJKi, IJKa -> ai", intermed["g_oooo_ββ"], intermed["rdm_ooov_ββ"])
            gen_fock_vo_β += einsum("IJKi, IJKa -> ai", intermed["g_oooo_αβ"], intermed["rdm_ooov_αβ"])

            gen_fock_oo_α += 0.5 * einsum("Ja bc, Ia bc -> IJ", intermed["g_ovvv_αα"], intermed["rdm_ovvv_αα"])
            gen_fock_oo_α += einsum("Ja bc, Ia bc -> IJ", intermed["g_ovvv_αβ"], intermed["rdm_ovvv_αβ"])
            gen_fock_oo_β += einsum("aJ bc, aI bc -> IJ", intermed["g_vovv_αβ"], intermed["rdm_vovv_αβ"])
            gen_fock_oo_β += 0.5 * einsum("Ja bc, Ia bc -> IJ", intermed["g_ovvv_ββ"], intermed["rdm_ovvv_ββ"])
            gen_fock_oo_α += einsum("Ji ja, Ii ja -> IJ", intermed["g_ooov_αα"], intermed["rdm_ooov_αα"])
            gen_fock_oo_α += einsum("Ji ja, Ii ja -> IJ", intermed["g_ooov_αβ"], intermed["rdm_ooov_αβ"])
            gen_fock_oo_α += einsum("Ji aj, Ii aj -> IJ", intermed["g_oovo_αβ"], intermed["rdm_oovo_αβ"])
            gen_fock_oo_β += einsum("iJ aj, iI aj -> IJ", intermed["g_oovo_αβ"], intermed["rdm_oovo_αβ"])
            gen_fock_oo_β += einsum("iJ ja, iI ja -> IJ", intermed["g_ooov_αβ"], intermed["rdm_ooov_αβ"])
            gen_fock_oo_β += einsum("Ji ja, Ii ja -> IJ", intermed["g_ooov_ββ"], intermed["rdm_ooov_ββ"])
            gen_fock_oo_α += 0.5 * einsum("ij Ja, ij Ia -> IJ", intermed["g_ooov_αα"], intermed["rdm_ooov_αα"])
            gen_fock_oo_α += einsum("ij Ja, ij Ia -> IJ", intermed["g_ooov_αβ"], intermed["rdm_ooov_αβ"])
            gen_fock_oo_β += einsum("ij aJ, ij aI -> IJ", intermed["g_oovo_αβ"], intermed["rdm_oovo_αβ"])
            gen_fock_oo_β += 0.5 * einsum("ij Ja, ij Ia -> IJ", intermed["g_ooov_ββ"], intermed["rdm_ooov_ββ"])

            gen_fock_vv_α += 0.5 * einsum("ij kB, ij kA -> AB", intermed["g_ooov_αα"], intermed["rdm_ooov_αα"])
            gen_fock_vv_α += einsum("ij Bk, ij Ak -> AB", intermed["g_oovo_αβ"], intermed["rdm_oovo_αβ"])
            gen_fock_vv_β += einsum("ij kB, ij kA -> AB", intermed["g_ooov_αβ"], intermed["rdm_ooov_αβ"])
            gen_fock_vv_β += 0.5 * einsum("ij kB, ij kA -> AB", intermed["g_ooov_ββ"], intermed["rdm_ooov_ββ"])
            gen_fock_vv_α += einsum("ia Bb, ia Ab -> AB", intermed["g_ovvv_αα"], intermed["rdm_ovvv_αα"])
            gen_fock_vv_α += einsum("ia Bb, ia Ab -> AB", intermed["g_ovvv_αβ"], intermed["rdm_ovvv_αβ"])
            gen_fock_vv_α += einsum("ai Bb, ai Ab -> AB", intermed["g_vovv_αβ"], intermed["rdm_vovv_αβ"])
            gen_fock_vv_β += einsum("ai bB, ai bA -> AB", intermed["g_vovv_αβ"], intermed["rdm_vovv_αβ"])
            gen_fock_vv_β += einsum("ia bB, ia bA -> AB", intermed["g_ovvv_αβ"], intermed["rdm_ovvv_αβ"])
            gen_fock_vv_β += einsum("ia bB, ia bA -> AB", intermed["g_ovvv_ββ"], intermed["rdm_ovvv_ββ"])
            gen_fock_vv_α += 0.5 * einsum("iB ab, iA ab -> AB", intermed["g_ovvv_αα"], intermed["rdm_ovvv_αα"])
            gen_fock_vv_α += einsum("Bi ab, Ai ab -> AB", intermed["g_vovv_αβ"], intermed["rdm_vovv_αβ"])
            gen_fock_vv_β += einsum("iB ab, iA ab -> AB", intermed["g_ovvv_αβ"], intermed["rdm_ovvv_αβ"])
            gen_fock_vv_β += 0.5 * einsum("iB ab, iA ab -> AB", intermed["g_ovvv_ββ"], intermed["rdm_ovvv_ββ"])

        gei = einsum("pq, Pp, Qq -> PQ", gen_fock_oo_α, orbitals["o"][0], orbitals["o"][0]) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_oo_β, orbitals["o"][1], orbitals["o"][1])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_vv_α, orbitals["v"][0], orbitals["v"][0])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_vv_β, orbitals["v"][1], orbitals["v"][1])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_ov_α, orbitals["o"][0], orbitals["v"][0])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_ov_β, orbitals["o"][1], orbitals["v"][1])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_vo_α, orbitals["v"][0], orbitals["o"][0])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_vo_β, orbitals["v"][1], orbitals["o"][1]))
        if "c_ooov_αα" in intermed:
            # We have some new TPDM terms
            tpdm += 2 * 4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ooov_αα"], orbitals["o"][0], orbitals["o"][0], orbitals["o"][0], orbitals["v"][0]) 
            tpdm += 8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ooov_αβ"], orbitals["o"][0], orbitals["o"][1], orbitals["o"][0], orbitals["v"][1]) 
            tpdm += 8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oovo_αβ"], orbitals["o"][0], orbitals["o"][1], orbitals["v"][0], orbitals["o"][1]) 
            tpdm += 2 * 4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ooov_ββ"], orbitals["o"][1], orbitals["o"][1], orbitals["o"][1], orbitals["v"][1]) 
            tpdm += 2 * 4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovvv_αα"], orbitals["o"][0], orbitals["v"][0], orbitals["v"][0], orbitals["v"][0]) 
            tpdm += 8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovvv_αβ"], orbitals["o"][0], orbitals["v"][1], orbitals["v"][0], orbitals["v"][1]) 
            tpdm += 8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vovv_αβ"], orbitals["v"][0], orbitals["o"][1], orbitals["v"][0], orbitals["v"][1]) 
            tpdm += 2 * 4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovvv_ββ"], orbitals["o"][1], orbitals["v"][1], orbitals["v"][1], orbitals["v"][1]) 

        # Construct alpha Gen. Fock
        # Construct beta Gen. Fock and back-transform
        for atom in range(len(GEOM)):
            hx_ao = program.core_hamiltonian_grad(BASIS, GEOM, atom)
            gx_ao = program.repulsion_grad(BASIS, GEOM, atom)
            sx_ao = program.overlap_grad(BASIS, GEOM, atom)
            for i, (h, g, s) in enumerate(zip(hx_ao, gx_ao, sx_ao)):
                grad[atom][i] = math_util.solvers.common.perturbation_gradient(gei, h, g, s, nx[atom][i], opdm, tpdm)
        print(grad)
        intermed["gradient"] = grad

    if test:
        # The dipole moment is the negative derivative of energy with respect to
        # electric field strength. Compare the analytic dipole moment (using the 1RDMs)
        # and the numerical dipole moment (finite difference of energies.)
        # If our implementation is correct, the two should match.
        p_ao = program.dipole(BASIS, GEOM)
        # TODO: This list should probably be pulled from elsewhere.
        if not hasattr(solver, "SI"):
            p = mla.request_asym(p_ao, orbitals, itertools.combinations_with_replacement(["c", "o", "v", "w"], 2))
            intermed["mu"] = 0
            for label, block in p.items():
                rdm_label = f"rdm_{label}"
                if rdm_label in intermed:
                    prefactor = 1 if label[0] == label[1] else 2
                    intermed["mu"] -= einsum("pq x, pq -> x", block, intermed[rdm_label]) * prefactor
        elif test:
            backtransformed_opdm = einsum("pq, Pp, Qq -> PQ", intermed["rdm_oo_α"], orbitals["o"][0], orbitals["o"][0]) + (
                einsum("pq, Pp, Qq -> PQ", intermed["rdm_oo_β"], orbitals["o"][1], orbitals["o"][1])) + (
                einsum("pq, Pp, Qq -> PQ", intermed["rdm_vv_α"], orbitals["v"][0], orbitals["v"][0])) + (
                einsum("pq, Pp, Qq -> PQ", intermed["rdm_vv_β"], orbitals["v"][1], orbitals["v"][1]))
            intermed["mu"] = -einsum("pq x, pq -> x", p_ao, backtransformed_opdm)
        intermed["deriv"] = math_util.hellmann_test(solver, h_ao, p_ao, r_ao, orbitals, r_thresh=1e-6)

    return intermed, orbitals

