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
        opdm = einsum("pq, Pp, Qq -> PQ", intermed["rdm_oo_??"], orbitals["o"][0], orbitals["o"][0]) + (
            einsum("pq, Pp, Qq -> PQ", intermed["rdm_oo_??"], orbitals["o"][1], orbitals["o"][1])) + (
            einsum("pq, Pp, Qq -> PQ", intermed["rdm_vv_??"], orbitals["v"][0], orbitals["v"][0])) + (
            einsum("pq, Pp, Qq -> PQ", intermed["rdm_vv_??"], orbitals["v"][1], orbitals["v"][1]))
        tpdm = 2 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oooo_????"], orbitals["o"][0], orbitals["o"][0], orbitals["o"][0], orbitals["o"][0]) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oooo_????"], orbitals["o"][0], orbitals["o"][1], orbitals["o"][0], orbitals["o"][1])) + (
            2 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oooo_????"], orbitals["o"][1], orbitals["o"][1], orbitals["o"][1], orbitals["o"][1])) + (
            2 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vvvv_????"], orbitals["v"][0], orbitals["v"][0], orbitals["v"][0], orbitals["v"][0])) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vvvv_????"], orbitals["v"][0], orbitals["v"][1], orbitals["v"][0], orbitals["v"][1])) + (
            2 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vvvv_????"], orbitals["v"][1], orbitals["v"][1], orbitals["v"][1], orbitals["v"][1])) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oovv_????"], orbitals["o"][0], orbitals["o"][0], orbitals["v"][0], orbitals["v"][0])) + (
            8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oovv_????"], orbitals["o"][0], orbitals["o"][1], orbitals["v"][0], orbitals["v"][1])) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oovv_????"], orbitals["o"][1], orbitals["o"][1], orbitals["v"][1], orbitals["v"][1])) + (
            4 * mla.antisymmetrize_axes_plus(
                einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovov_????"], orbitals["o"][0], orbitals["v"][0], orbitals["o"][0], orbitals["v"][0]),
                 ((0,), (1,)) )) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovov_????"], orbitals["o"][0], orbitals["v"][1], orbitals["o"][0], orbitals["v"][1])) + (
            4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vovo_????"], orbitals["v"][0], orbitals["o"][1], orbitals["v"][0], orbitals["o"][1])) + (
            8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovvo_????"], orbitals["o"][0], orbitals["v"][1], orbitals["v"][0], orbitals["o"][1])) + (
            4 * mla.antisymmetrize_axes_plus(
                einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovov_????"], orbitals["o"][1], orbitals["v"][1], orbitals["o"][1], orbitals["v"][1]),
                 ((0,), (1,)) ))
                
        gen_fock_oo_?? = einsum("i J, I i -> IJ", intermed["h_oo_??"], intermed["rdm_oo_??"]) + (
            0.5 * einsum("jk Ji, Ii jk -> IJ", intermed["g_oooo_????"], intermed["rdm_oooo_????"])) + (
            einsum("jk Ji, Ii jk -> IJ", intermed["g_oooo_????"], intermed["rdm_oooo_????"])) + (
            0.5 * einsum("Ji ab, Ii ab -> IJ", intermed["g_oovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("Ji ab, Ii ab -> IJ", intermed["g_oovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("ib Ja, Ia ib -> IJ", intermed["g_ovov_????"], intermed["rdm_ovov_????"])) + (
            einsum("ib Ja, Ia ib -> IJ", intermed["g_ovov_????"], intermed["rdm_ovov_????"])) + (
            einsum("Ja bi, Ia bi -> IJ", intermed["g_ovvo_????"], intermed["rdm_ovvo_????"]))
        gen_fock_oo_?? = einsum("i J, I i -> IJ", intermed["h_oo_??"], intermed["rdm_oo_??"]) + (
            0.5 * einsum("jk Ji, Ii jk -> IJ", intermed["g_oooo_????"], intermed["rdm_oooo_????"])) + (
            einsum("jk iJ, iI jk -> IJ", intermed["g_oooo_????"], intermed["rdm_oooo_????"])) + (
            0.5 * einsum("Ji ab, Ii ab -> IJ", intermed["g_oovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("iJ ab, iI ab -> IJ", intermed["g_oovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("ib Ja, Ia ib -> IJ", intermed["g_ovov_????"], intermed["rdm_ovov_????"])) + (
            einsum("bi aJ, aI bi -> IJ", intermed["g_vovo_????"], intermed["rdm_vovo_????"])) + (
            einsum("ib aJ, ib aI -> IJ", intermed["g_ovvo_????"], intermed["rdm_ovvo_????"]))
        gen_fock_vv_?? = einsum("a B, A a -> AB", intermed["h_vv_??"], intermed["rdm_vv_??"]) + (
            0.5 * einsum("bc Ba, Aa bc -> AB", intermed["g_vvvv_????"], intermed["rdm_vvvv_????"])) + (
            einsum("bc Ba, Aa bc -> AB", intermed["g_vvvv_????"], intermed["rdm_vvvv_????"])) + (
            0.5 * einsum("ij Ba, ij Aa -> AB", intermed["g_oovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("ij Ba, ij Aa -> AB", intermed["g_oovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("iB ja, iA ja -> AB", intermed["g_ovov_????"], intermed["rdm_ovov_????"])) + (
            einsum("Bi aj, Ai aj -> AB", intermed["g_vovo_????"], intermed["rdm_vovo_????"])) + (
            einsum("ja Bi, ja Ai -> AB", intermed["g_ovvo_????"], intermed["rdm_ovvo_????"]))
        gen_fock_vv_?? = einsum("a B, A a -> AB", intermed["h_vv_??"], intermed["rdm_vv_??"]) + (
            0.5 * einsum("bc Ba, Aa bc -> AB", intermed["g_vvvv_????"], intermed["rdm_vvvv_????"])) + (
            einsum("bc aB, aA bc -> AB", intermed["g_vvvv_????"], intermed["rdm_vvvv_????"])) + (
            0.5 * einsum("ij Ba, ij Aa -> AB", intermed["g_oovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("ij aB, ij aA -> AB", intermed["g_oovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("iB ja, iA ja -> AB", intermed["g_ovov_????"], intermed["rdm_ovov_????"])) + (
            einsum("iB ja, iA ja -> AB", intermed["g_ovov_????"], intermed["rdm_ovov_????"])) + (
            einsum("iB aj, iA aj -> AB", intermed["g_ovvo_????"], intermed["rdm_ovvo_????"]))
        gen_fock_ov_?? = einsum("Ia, Ii -> ia", intermed["h_ov_??"], intermed["rdm_oo_??"]) + (
            -0.5 * einsum("JK Ia, iI JK -> ia", intermed["g_ooov_????"], intermed["rdm_oooo_????"])) + (
            einsum("JK aI, iI JK -> ia", intermed["g_oovo_????"], intermed["rdm_oooo_????"])) + (
            -0.5 * einsum("Ia AB, iI AB -> ia", intermed["g_ovvv_????"], intermed["rdm_oovv_????"])) + (
            einsum("aI AB, iI AB -> ia", intermed["g_vovv_????"], intermed["rdm_oovv_????"])) + (
            einsum("IB aA, iA IB -> ia", intermed["g_ovvv_????"], intermed["rdm_ovov_????"])) + (
            einsum("IB aA, iA IB -> ia", intermed["g_ovvv_????"], intermed["rdm_ovov_????"])) + (
            einsum("BI aA, iA BI -> ia", intermed["g_vovv_????"], intermed["rdm_ovvo_????"]))
        gen_fock_ov_?? = einsum("Ia, Ii -> ia", intermed["h_ov_??"], intermed["rdm_oo_??"]) + (
            -0.5 * einsum("JK Ia, iI JK -> ia", intermed["g_ooov_????"], intermed["rdm_oooo_????"])) + (
            einsum("JK Ia, Ii JK -> ia", intermed["g_ooov_????"], intermed["rdm_oooo_????"])) + (
            -0.5 * einsum("Ia AB, iI AB -> ia", intermed["g_ovvv_????"], intermed["rdm_oovv_????"])) + (
            einsum("Ia AB, Ii AB -> ia", intermed["g_ovvv_????"], intermed["rdm_oovv_????"])) + (
            einsum("IB aA, iA IB -> ia", intermed["g_ovvv_????"], intermed["rdm_ovov_????"])) + (
            einsum("BI Aa, Ai BI -> ia", intermed["g_vovv_????"], intermed["rdm_vovo_????"])) + (
            einsum("IB Aa, IB Ai -> ia", intermed["g_ovvv_????"], intermed["rdm_ovvo_????"]))
        gen_fock_vo_?? = einsum("iA, Aa -> ai", intermed["h_ov_??"], intermed["rdm_vv_??"]) + (
            0.5 * einsum("iA BC, aA BC -> ai", intermed["g_ovvv_????"], intermed["rdm_vvvv_????"])) + (
            einsum("iA BC, aA BC -> ai", intermed["g_ovvv_????"], intermed["rdm_vvvv_????"])) + (
            0.5 * einsum("IJ iA, IJ aA -> ai", intermed["g_ooov_????"], intermed["rdm_oovv_????"])) + (
            einsum("IJ iA, IJ aA -> ai", intermed["g_ooov_????"], intermed["rdm_oovv_????"])) + (
            einsum("Ii JA, Ia JA -> ai", intermed["g_ooov_????"], intermed["rdm_ovov_????"])) + (
            einsum("iI JA, JA aI -> ai", intermed["g_ooov_????"], intermed["rdm_ovvo_????"])) + (
            einsum("iI AJ, aI AJ -> ai", intermed["g_oovo_????"], intermed["rdm_vovo_????"]))
        gen_fock_vo_?? = einsum("iA, Aa -> ai", intermed["h_ov_??"], intermed["rdm_vv_??"]) + (
            0.5 * einsum("iA BC, aA BC -> ai", intermed["g_ovvv_????"], intermed["rdm_vvvv_????"])) + (
            einsum("Ai CB, Aa CB -> ai", intermed["g_vovv_????"], intermed["rdm_vvvv_????"])) + (
            0.5 * einsum("IJ iA, IJ aA -> ai", intermed["g_ooov_????"], intermed["rdm_oovv_????"])) + (
            einsum("JI Ai, JI Aa -> ai", intermed["g_oovo_????"], intermed["rdm_oovv_????"])) + (
            einsum("Ii JA, Ia JA -> ai", intermed["g_ooov_????"], intermed["rdm_ovov_????"])) + (
            einsum("Ii JA, Ia JA -> ai", intermed["g_ooov_????"], intermed["rdm_ovov_????"])) + (
            einsum("Ii AJ, Ia AJ -> ai", intermed["g_oovo_????"], intermed["rdm_ovvo_????"]))
        if "rdm_ov_??" in intermed:
            opdm += 2 * einsum("pq, Pp, Qq -> PQ", intermed["rdm_ov_??"], orbitals["o"][0], orbitals["v"][0])
            opdm += 2 * einsum("pq, Pp, Qq -> PQ", intermed["rdm_ov_??"], orbitals["o"][1], orbitals["v"][1])
            gen_fock_oo_?? += einsum("J a, I a -> IJ", intermed["h_ov_??"], intermed["rdm_ov_??"])
            gen_fock_oo_?? += einsum("J a, I a -> IJ", intermed["h_ov_??"], intermed["rdm_ov_??"])
            gen_fock_ov_?? += einsum("A a, I a -> IA", intermed["h_vv_??"], intermed["rdm_ov_??"])
            gen_fock_ov_?? += einsum("A a, I a -> IA", intermed["h_vv_??"], intermed["rdm_ov_??"])
            gen_fock_vo_?? += einsum("I i, i A -> AI", intermed["h_oo_??"], intermed["rdm_ov_??"])
            gen_fock_vo_?? += einsum("I i, i A -> AI", intermed["h_oo_??"], intermed["rdm_ov_??"])
            gen_fock_vv_?? += einsum("i B, i A -> AB", intermed["h_ov_??"], intermed["rdm_ov_??"])
            gen_fock_vv_?? += einsum("i B, i A -> AB", intermed["h_ov_??"], intermed["rdm_ov_??"])
        if "rdm_ooov_????" in intermed:
            gen_fock_ov_?? +=  einsum("IaJA, IiJA-> ia", intermed["g_ovov_????"], intermed["rdm_ooov_????"])
            gen_fock_ov_?? +=  einsum("JAaI, iIJA-> ia", intermed["g_ovvo_????"], intermed["rdm_ooov_????"])
            gen_fock_ov_?? +=  einsum("aIAJ, iIAJ-> ia", intermed["g_vovo_????"], intermed["rdm_oovo_????"])
            gen_fock_ov_?? +=  einsum("IaJA, IiJA-> ia", intermed["g_ovov_????"], intermed["rdm_ooov_????"])
            gen_fock_ov_?? +=  einsum("IaJA, IiJA-> ia", intermed["g_ovov_????"], intermed["rdm_ooov_????"])
            gen_fock_ov_?? +=  einsum("IaAJ, IiAJ-> ia", intermed["g_ovvo_????"], intermed["rdm_oovo_????"])
            gen_fock_ov_?? += 0.5 * einsum("IJaA, IJiA -> ia", intermed["g_oovv_????"], intermed["rdm_ooov_????"])
            gen_fock_ov_?? += einsum("IJaA, IJiA -> ia", intermed["g_oovv_????"], intermed["rdm_ooov_????"])
            gen_fock_ov_?? += 0.5 * einsum("IJaA, IJiA -> ia", intermed["g_oovv_????"], intermed["rdm_ooov_????"])
            gen_fock_ov_?? += einsum("IJAa, IJAi -> ia", intermed["g_oovv_????"], intermed["rdm_oovo_????"])
            gen_fock_ov_?? += 0.5 * einsum("aABC, iABC-> ia", intermed["g_vvvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_ov_?? += einsum("aABC, iABC-> ia", intermed["g_vvvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_ov_?? += 0.5 * einsum("aABC, iABC-> ia", intermed["g_vvvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_ov_?? += einsum("AaBC, AiBC-> ia", intermed["g_vvvv_????"], intermed["rdm_vovv_????"])

            gen_fock_vo_?? += 0.5 * einsum("IiAB, IaAB -> ai", intermed["g_oovv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vo_?? += einsum("iIAB, aIAB -> ai", intermed["g_oovv_????"], intermed["rdm_vovv_????"])
            gen_fock_vo_?? += 0.5 * einsum("IiAB, IaAB -> ai", intermed["g_oovv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vo_?? += einsum("IiAB, IaAB -> ai", intermed["g_oovv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vo_?? += einsum("IAiB, IAaB -> ai", intermed["g_ovov_????"], intermed["rdm_ovvv_????"])
            gen_fock_vo_?? += einsum("IAiB, IAaB -> ai", intermed["g_ovov_????"], intermed["rdm_ovvv_????"])
            gen_fock_vo_?? += einsum("iBAI, AIaB -> ai", intermed["g_ovvo_????"], intermed["rdm_vovv_????"])
            gen_fock_vo_?? += einsum("IAiB, IAaB -> ai", intermed["g_ovov_????"], intermed["rdm_ovvv_????"])
            gen_fock_vo_?? += einsum("IABi, IABa -> ai", intermed["g_ovvo_????"], intermed["rdm_ovvv_????"])
            gen_fock_vo_?? += einsum("AIBi, AIBa -> ai", intermed["g_vovo_????"], intermed["rdm_vovv_????"])
            gen_fock_vo_?? += 0.5 * einsum("IJKi, IJKa -> ai", intermed["g_oooo_????"], intermed["rdm_ooov_????"])
            gen_fock_vo_?? += einsum("IJiK, IJaK -> ai", intermed["g_oooo_????"], intermed["rdm_oovo_????"])
            gen_fock_vo_?? += 0.5 * einsum("IJKi, IJKa -> ai", intermed["g_oooo_????"], intermed["rdm_ooov_????"])
            gen_fock_vo_?? += einsum("IJKi, IJKa -> ai", intermed["g_oooo_????"], intermed["rdm_ooov_????"])

            gen_fock_oo_?? += 0.5 * einsum("Ja bc, Ia bc -> IJ", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_oo_?? += einsum("Ja bc, Ia bc -> IJ", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_oo_?? += einsum("aJ bc, aI bc -> IJ", intermed["g_vovv_????"], intermed["rdm_vovv_????"])
            gen_fock_oo_?? += 0.5 * einsum("Ja bc, Ia bc -> IJ", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_oo_?? += einsum("Ji ja, Ii ja -> IJ", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_oo_?? += einsum("Ji ja, Ii ja -> IJ", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_oo_?? += einsum("Ji aj, Ii aj -> IJ", intermed["g_oovo_????"], intermed["rdm_oovo_????"])
            gen_fock_oo_?? += einsum("iJ aj, iI aj -> IJ", intermed["g_oovo_????"], intermed["rdm_oovo_????"])
            gen_fock_oo_?? += einsum("iJ ja, iI ja -> IJ", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_oo_?? += einsum("Ji ja, Ii ja -> IJ", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_oo_?? += 0.5 * einsum("ij Ja, ij Ia -> IJ", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_oo_?? += einsum("ij Ja, ij Ia -> IJ", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_oo_?? += einsum("ij aJ, ij aI -> IJ", intermed["g_oovo_????"], intermed["rdm_oovo_????"])
            gen_fock_oo_?? += 0.5 * einsum("ij Ja, ij Ia -> IJ", intermed["g_ooov_????"], intermed["rdm_ooov_????"])

            gen_fock_vv_?? += 0.5 * einsum("ij kB, ij kA -> AB", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_vv_?? += einsum("ij Bk, ij Ak -> AB", intermed["g_oovo_????"], intermed["rdm_oovo_????"])
            gen_fock_vv_?? += einsum("ij kB, ij kA -> AB", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_vv_?? += 0.5 * einsum("ij kB, ij kA -> AB", intermed["g_ooov_????"], intermed["rdm_ooov_????"])
            gen_fock_vv_?? += einsum("ia Bb, ia Ab -> AB", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vv_?? += einsum("ia Bb, ia Ab -> AB", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vv_?? += einsum("ai Bb, ai Ab -> AB", intermed["g_vovv_????"], intermed["rdm_vovv_????"])
            gen_fock_vv_?? += einsum("ai bB, ai bA -> AB", intermed["g_vovv_????"], intermed["rdm_vovv_????"])
            gen_fock_vv_?? += einsum("ia bB, ia bA -> AB", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vv_?? += einsum("ia bB, ia bA -> AB", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vv_?? += 0.5 * einsum("iB ab, iA ab -> AB", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vv_?? += einsum("Bi ab, Ai ab -> AB", intermed["g_vovv_????"], intermed["rdm_vovv_????"])
            gen_fock_vv_?? += einsum("iB ab, iA ab -> AB", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])
            gen_fock_vv_?? += 0.5 * einsum("iB ab, iA ab -> AB", intermed["g_ovvv_????"], intermed["rdm_ovvv_????"])

        gei = einsum("pq, Pp, Qq -> PQ", gen_fock_oo_??, orbitals["o"][0], orbitals["o"][0]) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_oo_??, orbitals["o"][1], orbitals["o"][1])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_vv_??, orbitals["v"][0], orbitals["v"][0])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_vv_??, orbitals["v"][1], orbitals["v"][1])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_ov_??, orbitals["o"][0], orbitals["v"][0])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_ov_??, orbitals["o"][1], orbitals["v"][1])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_vo_??, orbitals["v"][0], orbitals["o"][0])) + (
            einsum("pq, Pp, Qq -> PQ", gen_fock_vo_??, orbitals["v"][1], orbitals["o"][1]))
        if "c_ooov_????" in intermed:
            # We have some new TPDM terms
            tpdm += 2 * 4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ooov_????"], orbitals["o"][0], orbitals["o"][0], orbitals["o"][0], orbitals["v"][0]) 
            tpdm += 8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ooov_????"], orbitals["o"][0], orbitals["o"][1], orbitals["o"][0], orbitals["v"][1]) 
            tpdm += 8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_oovo_????"], orbitals["o"][0], orbitals["o"][1], orbitals["v"][0], orbitals["o"][1]) 
            tpdm += 2 * 4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ooov_????"], orbitals["o"][1], orbitals["o"][1], orbitals["o"][1], orbitals["v"][1]) 
            tpdm += 2 * 4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovvv_????"], orbitals["o"][0], orbitals["v"][0], orbitals["v"][0], orbitals["v"][0]) 
            tpdm += 8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovvv_????"], orbitals["o"][0], orbitals["v"][1], orbitals["v"][0], orbitals["v"][1]) 
            tpdm += 8 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_vovv_????"], orbitals["v"][0], orbitals["o"][1], orbitals["v"][0], orbitals["v"][1]) 
            tpdm += 2 * 4 * einsum("pqrs, Pp, Qq, Rr, Ss -> PQRS", intermed["rdm_ovvv_????"], orbitals["o"][1], orbitals["v"][1], orbitals["v"][1], orbitals["v"][1]) 

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
            backtransformed_opdm = einsum("pq, Pp, Qq -> PQ", intermed["rdm_oo_??"], orbitals["o"][0], orbitals["o"][0]) + (
                einsum("pq, Pp, Qq -> PQ", intermed["rdm_oo_??"], orbitals["o"][1], orbitals["o"][1])) + (
                einsum("pq, Pp, Qq -> PQ", intermed["rdm_vv_??"], orbitals["v"][0], orbitals["v"][0])) + (
                einsum("pq, Pp, Qq -> PQ", intermed["rdm_vv_??"], orbitals["v"][1], orbitals["v"][1]))
            intermed["mu"] = -einsum("pq x, pq -> x", p_ao, backtransformed_opdm)
        intermed["deriv"] = math_util.hellmann_test(solver, h_ao, p_ao, r_ao, orbitals, r_thresh=1e-6)

    return intermed, orbitals

