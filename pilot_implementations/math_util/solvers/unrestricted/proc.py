import numpy as np

from pilot_implementations.multilinear.tensor import einsum

def hermitian_block_orbital_gradient(i):

    ### OV
    # 1RDM part
    grad_α = 2 * einsum("Ia, Ii -> ia", i["h_ov_α"], i["rdm_oo_α"])
    grad_β = 2 * einsum("Ia, Ii -> ia", i["h_ov_β"], i["rdm_oo_β"])
    grad_α -= 2 * einsum("iA, Aa -> ia", i["h_ov_α"], i["rdm_vv_α"])
    grad_β -= 2 * einsum("iA, Aa -> ia", i["h_ov_β"], i["rdm_vv_β"])

    if "rdm_ov_α" in i:
        grad_α += 2 * einsum("Aa, iA -> ia", i["h_vv_α"], i["rdm_ov_α"])
        grad_β += 2 * einsum("Aa, iA -> ia", i["h_vv_β"], i["rdm_ov_β"])
        grad_α -= 2 * einsum("iI, Ia -> ia", i["h_oo_α"], i["rdm_ov_α"])
        grad_β -= 2 * einsum("iI, Ia -> ia", i["h_oo_β"], i["rdm_ov_β"])

    # 2RDM
    grad_α += -1 * einsum("JK Ia, iI JK -> ia", i["g_ooov_αα"], i["rdm_oooo_αα"])
    grad_α += +2 * einsum("JK aI, iI JK -> ia", i["g_oovo_αβ"], i["rdm_oooo_αβ"])
    grad_β += -1 * einsum("JK Ia, iI JK -> ia", i["g_ooov_ββ"], i["rdm_oooo_ββ"])
    grad_β += +2 * einsum("JK Ia, Ii JK -> ia", i["g_ooov_αβ"], i["rdm_oooo_αβ"])
    grad_α += -1 * einsum("Ia AB, iI AB -> ia", i["g_ovvv_αα"], i["rdm_oovv_αα"])
    grad_α += +2 * einsum("aI AB, iI AB -> ia", i["g_vovv_αβ"], i["rdm_oovv_αβ"])
    grad_β += -1 * einsum("Ia AB, iI AB -> ia", i["g_ovvv_ββ"], i["rdm_oovv_ββ"])
    grad_β += +2 * einsum("Ia AB, Ii AB -> ia", i["g_ovvv_αβ"], i["rdm_oovv_αβ"])
    grad_α +=  2 * einsum("IB aA, iA IB -> ia", i["g_ovvv_αα"], i["rdm_ovov_αα"])
    grad_α +=  2 * einsum("IB aA, iA IB -> ia", i["g_ovvv_αβ"], i["rdm_ovov_αβ"])
    grad_α +=  2 * einsum("BI aA, iA BI -> ia", i["g_vovv_αβ"], i["rdm_ovvo_αβ"])
    grad_β +=  2 * einsum("IB aA, iA IB -> ia", i["g_ovvv_ββ"], i["rdm_ovov_ββ"])
    grad_β +=  2 * einsum("BI Aa, Ai BI -> ia", i["g_vovv_αβ"], i["rdm_vovo_αβ"])
    grad_β +=  2 * einsum("IB Aa, IB Ai -> ia", i["g_ovvv_αβ"], i["rdm_ovvo_αβ"])

    grad_α += -1 * einsum("iA BC, aA BC -> ia", i["g_ovvv_αα"], i["rdm_vvvv_αα"])
    grad_α += -2 * einsum("iA BC, aA BC -> ia", i["g_ovvv_αβ"], i["rdm_vvvv_αβ"])
    grad_β += -1 * einsum("iA BC, aA BC -> ia", i["g_ovvv_ββ"], i["rdm_vvvv_ββ"])
    grad_β += -2 * einsum("Ai CB, Aa CB -> ia", i["g_vovv_αβ"], i["rdm_vvvv_αβ"])
    grad_α += -1 * einsum("IJ iA, IJ aA -> ia", i["g_ooov_αα"], i["rdm_oovv_αα"])
    grad_α += -2 * einsum("IJ iA, IJ aA -> ia", i["g_ooov_αβ"], i["rdm_oovv_αβ"])
    grad_β += -1 * einsum("IJ iA, IJ aA -> ia", i["g_ooov_ββ"], i["rdm_oovv_ββ"])
    grad_β += -2 * einsum("JI Ai, JI Aa -> ia", i["g_oovo_αβ"], i["rdm_oovv_αβ"])
    grad_α += -2 * einsum("Ii JA, Ia JA -> ia", i["g_ooov_αα"], i["rdm_ovov_αα"])
    grad_α += -2 * einsum("iI JA, JA aI -> ia", i["g_ooov_αβ"], i["rdm_ovvo_αβ"])
    grad_α += -2 * einsum("iI AJ, aI AJ -> ia", i["g_oovo_αβ"], i["rdm_vovo_αβ"])
    grad_β += -2 * einsum("Ii JA, Ia JA -> ia", i["g_ooov_ββ"], i["rdm_ovov_ββ"])
    grad_β += -2 * einsum("Ii JA, Ia JA -> ia", i["g_ooov_αβ"], i["rdm_ovov_αβ"])
    grad_β += -2 * einsum("Ii AJ, Ia AJ -> ia", i["g_oovo_αβ"], i["rdm_ovvo_αβ"])

    if "rdm_ooov_αα" in i:
        grad_α += 2 * einsum("IaJA, IiJA-> ia", i["g_ovov_αα"], i["rdm_ooov_αα"])
        grad_α += 2 * einsum("JAaI, iIJA-> ia", i["g_ovvo_αβ"], i["rdm_ooov_αβ"])
        grad_α += 2 * einsum("aIAJ, iIAJ-> ia", i["g_vovo_αβ"], i["rdm_oovo_αβ"])
        grad_β += 2 * einsum("IaJA, IiJA-> ia", i["g_ovov_ββ"], i["rdm_ooov_ββ"])
        grad_β += 2 * einsum("IaJA, IiJA-> ia", i["g_ovov_αβ"], i["rdm_ooov_αβ"])
        grad_β += 2 * einsum("IaAJ, IiAJ-> ia", i["g_ovvo_αβ"], i["rdm_oovo_αβ"])
        grad_α += einsum("IJaA, IJiA -> ia", i["g_oovv_αα"], i["rdm_ooov_αα"])
        grad_α += 2 * einsum("IJaA, IJiA -> ia", i["g_oovv_αβ"], i["rdm_ooov_αβ"])
        grad_β += einsum("IJaA, IJiA -> ia", i["g_oovv_ββ"], i["rdm_ooov_ββ"])
        grad_β += 2 * einsum("IJAa, IJAi -> ia", i["g_oovv_αβ"], i["rdm_oovo_αβ"])
        grad_α += einsum("aABC, iABC-> ia", i["g_vvvv_αα"], i["rdm_ovvv_αα"])
        grad_α += 2 * einsum("aABC, iABC-> ia", i["g_vvvv_αβ"], i["rdm_ovvv_αβ"])
        grad_β += einsum("aABC, iABC-> ia", i["g_vvvv_ββ"], i["rdm_ovvv_ββ"])
        grad_β += 2 * einsum("AaBC, AiBC-> ia", i["g_vvvv_αβ"], i["rdm_vovv_αβ"])

        grad_α -= einsum("IiAB, IaAB -> ia", i["g_oovv_αα"], i["rdm_ovvv_αα"])
        grad_α -= 2 * einsum("iIAB, aIAB -> ia", i["g_oovv_αβ"], i["rdm_vovv_αβ"])
        grad_β -= einsum("IiAB, IaAB -> ia", i["g_oovv_ββ"], i["rdm_ovvv_ββ"])
        grad_β -= 2 * einsum("IiAB, IaAB -> ia", i["g_oovv_αβ"], i["rdm_ovvv_αβ"])
        grad_α -= 2 * einsum("IAiB, IAaB -> ia", i["g_ovov_αα"], i["rdm_ovvv_αα"])
        grad_α -= 2 * einsum("IAiB, IAaB -> ia", i["g_ovov_αβ"], i["rdm_ovvv_αβ"])
        grad_α -= 2 * einsum("iBAI, AIaB -> ia", i["g_ovvo_αβ"], i["rdm_vovv_αβ"])
        grad_β -= 2 * einsum("IAiB, IAaB -> ia", i["g_ovov_ββ"], i["rdm_ovvv_ββ"])
        grad_β -= 2 * einsum("IABi, IABa -> ia", i["g_ovvo_αβ"], i["rdm_ovvv_αβ"])
        grad_β -= 2 * einsum("AIBi, AIBa -> ia", i["g_vovo_αβ"], i["rdm_vovv_αβ"])
        grad_α -= einsum("IJKi, IJKa -> ia", i["g_oooo_αα"], i["rdm_ooov_αα"])
        grad_α -= 2 * einsum("IJiK, IJaK -> ia", i["g_oooo_αβ"], i["rdm_oovo_αβ"])
        grad_β -= einsum("IJKi, IJKa -> ia", i["g_oooo_ββ"], i["rdm_ooov_ββ"])
        grad_β -= 2 * einsum("IJKi, IJKa -> ia", i["g_oooo_αβ"], i["rdm_ooov_αβ"])

    i["r1_ov_α"] = grad_α
    i["r1_ov_β"] = grad_β

def compute_integrals(intermediates, h_ao, r_ao, orbitals): 
        Coa, Cob = orbitals["o"]
        Cva, Cvb = orbitals["v"]

        intermediates["h_oo_α"] = einsum("pq, pP, qQ -> PQ", h_ao, Coa, Coa)
        intermediates["h_oo_β"] = einsum("pq, pP, qQ -> PQ", h_ao, Cob, Cob)
        intermediates["h_ov_α"] = einsum("pq, pP, qQ -> PQ", h_ao, Coa, Cva)
        intermediates["h_ov_β"] = einsum("pq, pP, qQ -> PQ", h_ao, Cob, Cvb)
        intermediates["h_vv_α"] = einsum("pq, pP, qQ -> PQ", h_ao, Cva, Cva)
        intermediates["h_vv_β"] = einsum("pq, pP, qQ -> PQ", h_ao, Cvb, Cvb)


        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Coa, Coa, Coa)
        temp -= np.swapaxes(temp, 2, 3)
        intermediates["g_oooo_αα"] = temp
        intermediates["g_oooo_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cob, Coa, Cob)
        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cob, Cob, Cob, Cob)
        temp -= np.swapaxes(temp, 2, 3)
        intermediates["g_oooo_ββ"] = temp

        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Coa, Cva, Cva)
        temp -= np.swapaxes(temp, 2, 3)
        intermediates["g_oovv_αα"] = temp
        intermediates["g_oovv_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cob, Cva, Cvb)
        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cob, Cob, Cvb, Cvb)
        temp -= np.swapaxes(temp, 2, 3)
        intermediates["g_oovv_ββ"] = temp

        intermediates["g_vvoo_αα"] = einsum("PQRS -> RSPQ", intermediates["g_oovv_αα"])
        intermediates["g_vvoo_αβ"] = einsum("PQRS -> RSPQ", intermediates["g_oovv_αβ"])
        intermediates["g_vvoo_ββ"] = einsum("PQRS -> RSPQ", intermediates["g_oovv_ββ"])
        
        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cva, Cva, Cva, Cva)
        temp -= np.swapaxes(temp, 2, 3)
        intermediates["g_vvvv_αα"] = temp
        intermediates["g_vvvv_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cva, Cvb, Cva, Cvb)
        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cvb, Cvb, Cvb, Cvb)
        temp -= np.swapaxes(temp, 2, 3)
        intermediates["g_vvvv_ββ"] = temp

        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cva, Coa, Cva)
        temp -= einsum("pqrs, pP, qQ, rS, sR -> PQ RS", r_ao, Coa, Cva, Cva, Coa)
        intermediates["g_ovov_αα"] = temp
        intermediates["g_ovov_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cvb, Coa, Cvb)
        intermediates["g_vovo_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cva, Cob, Cva, Cob)
        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cob, Cvb, Cob, Cvb)
        temp -= einsum("pqrs, pP, qQ, rS, sR -> PQ RS", r_ao, Cob, Cvb, Cvb, Cob)
        intermediates["g_ovov_ββ"] = temp
        intermediates["g_ovvo_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cvb, Cva, Cob)

        intermediates["g_voov_αβ"] = einsum("PQRS -> RSPQ", intermediates["g_ovvo_αβ"])

        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Coa, Coa, Cva)
        temp -= np.swapaxes(temp, 0, 1)
        intermediates["g_ooov_αα"] = temp
        intermediates["g_ovoo_αα"] = einsum("PQRS -> RSPQ", intermediates["g_ooov_αα"])
        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cob, Cob, Cob, Cvb)
        temp -= np.swapaxes(temp, 0, 1)
        intermediates["g_ooov_ββ"] = temp
        intermediates["g_ovoo_ββ"] = einsum("PQRS -> RSPQ", intermediates["g_ooov_ββ"])
        intermediates["g_oovo_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cob, Cva, Cob)
        intermediates["g_vooo_αβ"] = einsum("PQRS -> RSPQ", intermediates["g_oovo_αβ"])
        intermediates["g_ooov_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cob, Coa, Cvb)
        intermediates["g_ovoo_αβ"] = einsum("PQRS -> RSPQ", intermediates["g_ooov_αβ"])

        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cva, Cva, Cva)
        temp -= np.swapaxes(temp, 2, 3)
        intermediates["g_ovvv_αα"] = temp
        intermediates["g_vvov_αα"] = einsum("PQRS -> RSPQ", intermediates["g_ovvv_αα"])
        temp = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cob, Cvb, Cvb, Cvb)
        temp -= np.swapaxes(temp, 2, 3)
        intermediates["g_ovvv_ββ"] = temp
        intermediates["g_vvov_ββ"] = einsum("PQRS -> RSPQ", intermediates["g_ovvv_ββ"])
        intermediates["g_vovv_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Cva, Cob, Cva, Cvb)
        intermediates["g_vvvo_αβ"] = einsum("PQRS -> RSPQ", intermediates["g_vovv_αβ"])
        intermediates["g_ovvv_αβ"] = einsum("pqrs, pP, qQ, rR, sS -> PQ RS", r_ao, Coa, Cvb, Cva, Cvb)
        intermediates["g_vvov_αβ"] = einsum("PQRS -> RSPQ", intermediates["g_ovvv_αβ"])


