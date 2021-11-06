from functools import partial

from pilot_implementations import multilinear as mla
from pilot_implementations.multilinear.tensor import einsum
import numpy as np
from .. import common

def build_intermed(connected, opdm, opdm_product, intermed, ov=False):
    zero_rdms(intermed, ov)
    connected(intermed)
    opdm(intermed)
    opdm_product(intermed)
    assemble_fock_block_diagonal(intermed)
    rdm_construct(intermed)

def build_intermed_SI(connected, opdm, opdm_product, intermed, ov=False):
    zero_rdms_SI(intermed, ov)
    connected(intermed)
    opdm(intermed)
    opdm_product(intermed)
    assemble_fock_block_diagonal_SI(intermed)
    rdm_construct_SI(intermed)

def zero_rdms(inter, ov):
    strings = {"oovv", "oooo", "vvvv", "ovov", "oo", "vv"}
    if ov is True:
        strings.update(["ov", "ooov", "ovvv"])
    elif ov == "half":
        strings.update(["ooov", "ovvv"])
    elif ov is not False:
        raise Exception

    for string in strings:
        inter[f"rdm_{string}"] = 0
        inter[f"c_{string}"] = 0

def zero_rdms_SI(inter, ov):
    strings = {
            "oovv_αα", "oovv_αβ", "oovv_ββ", "oooo_αα",
            "oooo_αβ", "oooo_ββ", "vvvv_αα", "vvvv_αβ",
            "vvvv_ββ", "ovov_αα", "ovov_αβ", "ovvo_αβ",
            "vovo_αβ", "ovov_ββ", "oo_α", "oo_β",
            "vv_α", "vv_β"}

    if ov is True:
        strings.update([
            "ov_α", "ov_β",
            "ooov_αα", "ooov_αβ", "oovo_αβ", "ooov_ββ",
            "ovvv_αα", "ovvv_αβ", "vovv_αβ", "ovvv_ββ"])
    elif ov == "half":
        strings.update([
            "ooov_αα", "ooov_αβ", "oovo_αβ", "ooov_ββ",
            "ovvv_αα", "ovvv_αβ", "vovv_αβ", "ovvv_ββ"])
    elif ov is not False:
        raise Exception

    for string in strings:
        inter[f"rdm_{string}"] = 0
        inter[f"c_{string}"] = 0

def rdm_construct(inter):
    kappa = np.eye(inter["rdm_oo"].shape[0])
    inter["rdm_oovv"] += inter["c_oovv"]
    inter["rdm_oooo"] += inter["c_oooo"]
    inter["rdm_oooo"] += mla.antisymmetrize_axes(einsum("pr, qs -> pqrs", kappa, kappa), (2, 3))
    inter["rdm_oooo"] += mla.antisymmetrize_axes(einsum("pr, qs -> pqrs", kappa, inter["rdm_oo"]), (0, 1), (2, 3))
    inter["rdm_vvvv"] += inter["c_vvvv"]
    inter["rdm_ovov"] += inter["c_ovov"] + einsum("p r, q s -> pqrs", kappa, inter["rdm_vv"])
    inter["rdm_oo"] += kappa
    if "c_ooov" in inter:
        inter["rdm_ooov"] += inter["c_ooov"]
        if "rdm_ov" in inter:
            inter["rdm_ooov"] += mla.antisymmetrize_axes(einsum("p r, q s -> pqrs", kappa, inter["rdm_ov"]), (0, 1))
        inter["rdm_ovvv"] += inter["c_ovvv"]

def rdm_construct_SI(inter):
    kappa_a = np.eye(inter["rdm_oo_α"].shape[0])
    kappa_b = np.eye(inter["rdm_oo_β"].shape[0])
    inter["rdm_oovv_αα"] += inter["c_oovv_αα"]
    inter["rdm_oovv_αβ"] += inter["c_oovv_αβ"]
    inter["rdm_oovv_ββ"] += inter["c_oovv_ββ"]
    inter["rdm_oooo_αα"] += inter["c_oooo_αα"]
    inter["rdm_oooo_αβ"] += inter["c_oooo_αβ"]
    inter["rdm_oooo_ββ"] += inter["c_oooo_ββ"]
    inter["rdm_oooo_αα"] += mla.antisymmetrize_axes(einsum("pr, qs -> pqrs", kappa_a, kappa_a), (2, 3))
    inter["rdm_oooo_αβ"] += einsum("pr, qs -> pqrs", kappa_a, kappa_b)
    inter["rdm_oooo_ββ"] += mla.antisymmetrize_axes(einsum("pr, qs -> pqrs", kappa_b, kappa_b), (2, 3))
    inter["rdm_oooo_αα"] += mla.antisymmetrize_axes(einsum("pr, qs -> pqrs", kappa_a, inter["rdm_oo_α"]), (0, 1), (2, 3))
    inter["rdm_oooo_αβ"] += einsum("pr, qs -> pqrs", inter["rdm_oo_α"], kappa_b)
    inter["rdm_oooo_αβ"] += einsum("pr, qs -> pqrs", kappa_a, inter["rdm_oo_β"])
    inter["rdm_oooo_ββ"] += mla.antisymmetrize_axes(einsum("pr, qs -> pqrs", kappa_b, inter["rdm_oo_β"]), (0, 1), (2, 3))
    inter["rdm_vvvv_αα"] += inter["c_vvvv_αα"]
    inter["rdm_vvvv_αβ"] += inter["c_vvvv_αβ"]
    inter["rdm_vvvv_ββ"] += inter["c_vvvv_ββ"]
    inter["rdm_ovov_αα"] += inter["c_ovov_αα"] + einsum("p r, q s -> pqrs", kappa_a, inter["rdm_vv_α"])
    inter["rdm_ovov_αβ"] += inter["c_ovov_αβ"] + einsum("p r, q s -> pqrs", kappa_a, inter["rdm_vv_β"])
    inter["rdm_ovvo_αβ"] += inter["c_ovvo_αβ"]
    inter["rdm_vovo_αβ"] += inter["c_vovo_αβ"] + einsum("p r, q s -> pqrs", inter["rdm_vv_α"], kappa_b)
    inter["rdm_ovov_ββ"] += inter["c_ovov_ββ"] + einsum("p r, q s -> pqrs", kappa_b, inter["rdm_vv_β"])
    inter["rdm_oo_α"] += kappa_a
    inter["rdm_oo_β"] += kappa_b
    if "c_ooov_αα" in inter:
        inter["rdm_ooov_αα"] += inter["c_ooov_αα"]
        inter["rdm_ooov_αβ"] += inter["c_ooov_αβ"]
        inter["rdm_oovo_αβ"] += inter["c_oovo_αβ"]
        inter["rdm_ooov_ββ"] += inter["c_ooov_ββ"]
        if "rdm_ov_α" in inter:
            inter["rdm_ooov_αα"] += mla.antisymmetrize_axes(einsum("p r, q s -> pqrs", kappa_a, inter["rdm_ov_α"]), (0, 1))
            inter["rdm_ooov_αβ"] += einsum("p r, q s -> pqrs", kappa_a, inter["rdm_ov_β"])
            inter["rdm_oovo_αβ"] += einsum("p r, q s -> pqrs", inter["rdm_ov_α"], kappa_b)
            inter["rdm_ooov_ββ"] += mla.antisymmetrize_axes(einsum("p r, q s -> pqrs", kappa_b, inter["rdm_ov_β"]), (0, 1))
        inter["rdm_ovvv_αα"] += inter["c_ovvv_αα"]
        inter["rdm_ovvv_αβ"] += inter["c_ovvv_αβ"]
        inter["rdm_vovv_αβ"] += inter["c_vovv_αβ"]
        inter["rdm_ovvv_ββ"] += inter["c_ovvv_ββ"]

def assemble_fock_block_diagonal(inter):
    inter["f_oo"] = inter["h_oo"] + einsum("Ii Ji -> IJ", inter["g_oooo"])
    inter["f_vv"] = inter["h_vv"] + einsum("iA iB -> AB", inter["g_ovov"])
    if "rdm_ov" in inter:
        inter["f_ov"] = inter["h_ov"] + einsum("iI iA -> IA", inter["g_ooov"])

def assemble_fock_block_diagonal_SI(inter):
    inter["f_oo_α"] = inter["h_oo_α"] + einsum("Ii Ji -> IJ", inter["g_oooo_αα"]) + einsum("Ii Ji -> IJ", inter["g_oooo_αβ"])
    inter["f_oo_β"] = inter["h_oo_β"] + einsum("Ii Ji -> IJ", inter["g_oooo_ββ"]) + einsum("iI iJ -> IJ", inter["g_oooo_αβ"])
    inter["f_vv_α"] = inter["h_vv_α"] + einsum("iA iB -> AB", inter["g_ovov_αα"]) + einsum("Ai Bi -> AB", inter["g_vovo_αβ"])
    inter["f_vv_β"] = inter["h_vv_β"] + einsum("iA iB -> AB", inter["g_ovov_ββ"]) + einsum("iA iB -> AB", inter["g_ovov_αβ"])
    if "rdm_ov_α" in inter:
        inter["f_ov_α"] = inter["h_ov_α"] + einsum("iI iA -> IA", inter["g_ooov_αα"]) + einsum("Ii Ai -> IA", inter["g_oovo_αβ"])
        inter["f_ov_β"] = inter["h_ov_β"] + einsum("iI iA -> IA", inter["g_ooov_ββ"]) + einsum("iI iA -> IA", inter["g_ooov_αβ"])

diagonal_dict = {
        "o": lambda x: np.diagonal(x["f_oo"]),
        "v": lambda x: - np.diagonal(x["f_vv"]),
}

simultaneous_step = partial(common.simultaneous_step, diagonal_dict = diagonal_dict)

diagonal_spin_dict = {
        "oα": lambda x: np.diagonal(x["f_oo_α"]),
        "vα": lambda x: - np.diagonal(x["f_vv_α"]),
        "oβ": lambda x: np.diagonal(x["f_oo_β"]),
        "vβ": lambda x: - np.diagonal(x["f_vv_β"])
        }

simultaneous_step_SI = partial(common.simultaneous_step_SI, diagonal_dict = diagonal_spin_dict)

