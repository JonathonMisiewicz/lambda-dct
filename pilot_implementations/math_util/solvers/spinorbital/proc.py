import numpy as np

from pilot_implementations import multilinear as mla
from pilot_implementations.multilinear.tensor import einsum

def hermitian_block_orbital_gradient(i):
    def make_unit(mat):
        return np.eye(mat.shape[0])

    ### OV
    # 1RDM part
    grad = 2 * einsum("Ia, Ii -> ia", i["h_ov"], i["rdm_oo"])
    grad -= 2 * einsum("iA, Aa -> ia", i["h_ov"], i["rdm_vv"])
    # 2RDM
    grad += -1 * einsum("JK Ia, iI JK -> ia", i["g_ooov"], i["rdm_oooo"])
    grad += -1 * einsum("Ia AB, iI AB -> ia", i["g_ovvv"], i["rdm_oovv"])
    grad +=  2 * einsum("IB aA, iA IB -> ia", i["g_ovvv"], i["rdm_ovov"])

    grad += -1 * einsum("iA BC, aA BC -> ia", i["g_ovvv"], i["rdm_vvvv"])
    grad += -1 * einsum("IJ iA, IJ aA -> ia", i["g_ooov"], i["rdm_oovv"])
    grad += -2 * einsum("Ii JA, Ia JA -> ia", i["g_ooov"], i["rdm_ovov"])

    if "rdm_ov" in i:
        grad += 2 * einsum("aA, iA -> ia", i["h_vv"], i["rdm_ov"])
        grad -= 2 * einsum("iI, Ia -> ia", i["h_oo"], i["rdm_ov"])

    if "rdm_ooov" in i:
        grad += 2 * einsum("IaJA, IiJA-> ia", i["g_ovov"], i["rdm_ooov"])
        grad += einsum("IJaA, IJiA -> ia", i["g_oovv"], i["rdm_ooov"])
        grad += einsum("aABC, iABC-> ia", i["g_vvvv"], i["rdm_ovvv"])

        grad -= einsum("IiAB, IaAB -> ia", i["g_oovv"], i["rdm_ovvv"])
        grad -= 2 * einsum("IAiB, IAaB -> ia", i["g_ovov"], i["rdm_ovvv"])
        grad -= einsum("IJKi, IJKa -> ia", i["g_oooo"], i["rdm_ooov"])

    i["r1ov"] = grad

def backtransform_hermitian_opdm(orbitals, intermed):
    """ Assume that the 1RDM is diagonal in the occupied block, has an occ-occ and vir-vir block, """
    ci = orbitals["o"]
    cv = orbitals["v"]
    oei = mla.mso_to_aso(intermed["rdm_oo"], (ci, ci))
    if "rdm_ov" in intermed:
        oei += 2 * mla.mso_to_aso(intermed["rdm_ov"], (ci, cv))
    oei += mla.mso_to_aso(intermed["rdm_vv"], (cv, cv))
    return oei 

def backtransform_hermitian_tpdm(orbitals, intermed):
    ci = orbitals["o"]
    cv = orbitals["v"]
    # oovv; vvoo
    tei = 2 * mla.mso_to_aso(intermed["rdm_oovv"], (ci, ci, cv, cv))
    # ovov; voov; ovvo; vovo
    tei += 4 * mla.mso_to_aso(intermed["rdm_ovov"], (ci, cv, ci, cv))
    tei += mla.mso_to_aso(intermed["rdm_oooo"], (ci, ci, ci, ci))
    if "rdm_vvvv" in intermed: # False for OMP2
        tei += mla.mso_to_aso(intermed["rdm_vvvv"], (cv, cv, cv, cv))
    if "rdm_ovvv" in intermed: # False for OMP2, OCEPA, and OUDCT doubles
        tei += 4 * mla.mso_to_aso(intermed["rdm_ovvv"], (ci, cv, cv, cv))
        tei += 4 * mla.mso_to_aso(intermed["rdm_ooov"], (ci, ci, ci, cv))
    return tei

def backtransform_hermitian_gfm(orbitals, intermed):
    # Backtransform perpendicular rotation derivative over 2
    # The division by two corrects for double-counting constraints
    # We assume no frozen orbitals, the GFM is hermitian, and no rdm_ov. Beyond that, the method tells us what to do.
    ci = orbitals["o"]
    cv = orbitals["v"]
    nina = ci[0].shape[1] + ci[1].shape[1]
    nvir = cv[0].shape[1] + cv[1].shape[1]
    gei = np.zeros((nina + nvir, nina + nvir))
    # OO block:
    term = einsum("i J, I i -> IJ", intermed["h_oo"], intermed["rdm_oo"])
    term += 0.5 * einsum("jk Ji, Ii jk -> IJ", intermed["g_oooo"], intermed["rdm_oooo"])
    term += 0.5 * einsum("Ji ab, Ii ab -> IJ", intermed["g_oovv"], intermed["rdm_oovv"])
    term += einsum("ib Ja, Ia ib -> IJ", intermed["g_ovov"], intermed["rdm_ovov"])
    if "rdm_ooov" in intermed:
        term += 0.5 * einsum("Ja bc, Ia bc -> IJ", intermed["g_ovvv"], intermed["rdm_ovvv"])
        term += einsum("Ji ja, Ii ja -> IJ", intermed["g_ooov"], intermed["rdm_ooov"])
        term += 0.5 * einsum("ij Ja, ij Ia -> IJ", intermed["g_ooov"], intermed["rdm_ooov"])
    if "rdm_ov" in intermed:
        term += einsum("J a, I a -> IJ", intermed["h_ov"], intermed["rdm_ov"])
    gei += mla.mso_to_aso(term, (ci, ci))
    # OV block (Should equal VO):
    term = einsum("i A, I i -> IA", intermed["h_ov"], intermed["rdm_oo"])
    term += -0.5 * einsum("jk iA, Ii jk -> IA", intermed["g_ooov"], intermed["rdm_oooo"])
    term += -0.5 * einsum("iA ab, Ii ab -> IA", intermed["g_ovvv"], intermed["rdm_oovv"])
    term += -einsum("ib aA, Ia ib -> IA", intermed["g_ovvv"], intermed["rdm_ovov"])
    if "rdm_ooov" in intermed:
        term += 0.5 * einsum("Aa bc, Ia bc -> IA", intermed["g_vvvv"], intermed["rdm_ovvv"])
        term += 0.5 * einsum("ij Aa, ij Ia -> IA", intermed["g_oovv"], intermed["rdm_ooov"])
        term += einsum("iA ja, iI ja -> IA", intermed["g_ovov"], intermed["rdm_ooov"])
    if "rdm_ov" in intermed:
        term += einsum("a A, I a -> IA", intermed["h_vv"], intermed["rdm_ov"])
    gei += mla.mso_to_aso(term, (ci, cv))
    # VO block (Should equal OV):
    term = einsum("I a, A a -> AI", intermed["h_ov"], intermed["rdm_vv"])
    term += 0.5 * einsum("Ia bc, Aa bc -> AI", intermed["g_ovvv"], intermed["rdm_vvvv"])
    term += 0.5 * einsum("ij Ia, ij Aa -> AI", intermed["g_ooov"], intermed["rdm_oovv"])
    term += einsum("iI ja, iA ja -> AI", intermed["g_ooov"], intermed["rdm_ovov"])
    if "rdm_ooov" in intermed:
        term += 0.5 * einsum("ij kI, ij kA -> AI", intermed["g_oooo"], intermed["rdm_ooov"])
        term += 0.5 * einsum("iI ab, iA ab -> AI", intermed["g_oovv"], intermed["rdm_ovvv"])
        term += einsum("ia Ib, ia Ab -> AI", intermed["g_ovov"], intermed["rdm_ovvv"])
    if "rdm_ov" in intermed:
        term += einsum("I i, i A -> AI", intermed["h_oo"], intermed["rdm_ov"])
    gei += mla.mso_to_aso(term, (cv, ci))
    # VV block:
    term = einsum("a B, A a -> AB", intermed["h_vv"], intermed["rdm_vv"])
    term += 0.5 * einsum("bc Ba, Aa bc -> AB", intermed["g_vvvv"], intermed["rdm_vvvv"])
    term += 0.5 * einsum("ij Ba, ij Aa -> AB", intermed["g_oovv"], intermed["rdm_oovv"])
    term += einsum("iB ja, iA ja -> AB", intermed["g_ovov"], intermed["rdm_ovov"])
    if "rdm_ovvv" in intermed:
        term += 0.5 * einsum("ij kB, ij kA -> AB", intermed["g_ooov"], intermed["rdm_ooov"])
        term += 0.5 * einsum("iB ab, iA ab -> AB", intermed["g_ovvv"], intermed["rdm_ovvv"])
        term += einsum("ia Bb, ia Ab -> AB", intermed["g_ovvv"], intermed["rdm_ovvv"])
    if "rdm_ov" in intermed:
        term += einsum("i B, i A -> AB", intermed["h_ov"], intermed["rdm_ov"])
    gei += mla.mso_to_aso(term, (cv, cv))

    return gei 

