from pilot_implementations.multilinear.tensor import einsum
from pilot_implementations import multilinear as mla
from pilot_implementations.math_util.convergence import DirectSumDiis
import numpy as np
import re
from math import factorial
import collections as col

def simultaneous_step(inter, diagonal_dict):
    # Identify all strings with a t.
    strings = [key[1:] for key in inter if key.startswith("t")]
    # Take Jacobi step
    for string in strings:
        rank = int(re.match(r"\d*", string).group(0))
        if rank == 1 and len(string) != 1:
            left = diagonal_dict[string[1]](inter)
            right = diagonal_dict[string[2]](inter)
        else:
            left = diagonal_dict["o"](inter)
            right = diagonal_dict["v"](inter)
        hess = 2 * mla.broadcaster(rank, left, right)
        inter[f"t{string}"] += inter[f"r{string}"] / hess

    new_amps = inter["dsd"].diis(tuple(inter[f"r{string}"] for string in strings),
                        tuple(inter[f"t{string}"] for string in strings))
    for string, t in zip(strings, new_amps):
        inter[f"t{string}"] = t

def simultaneous_step_SI(inter, diagonal_dict):
    strings = set(key[1:] for key in inter if key.startswith("t"))
    for string in strings:
        spin = string.split("_")[-1]
        axes = [diagonal_dict["o" + x](inter) for x in spin] + [diagonal_dict["v" + x](inter) for x in spin]
        hess = 2 * mla.full_broadcaster(axes)
        inter[f"t{string}"] += inter[f"r{string}"] / hess

    new_amps = inter["dsd"].diis(tuple(inter[f"r{string}"] for string in strings),
                        tuple(inter[f"t{string}"] for string in strings))
    for string, t in zip(strings, new_amps):
        inter[f"t{string}"] = t


def initialize_intermediates_hf(orbitals, ranks=[2]):
    ncor = sum(x.shape[1] for x in orbitals["c"])
    nocc = sum(x.shape[1] for x in orbitals["o"])
    nvir = sum(x.shape[1] for x in orbitals["v"])
    nfrz = sum(x.shape[1] for x in orbitals["w"])
    intermed = {"dsd": DirectSumDiis(3, 9)}
    for rank in ranks:
        intermed[f"t{rank}"] = mla.read_tensor(f"t{rank}", rank, nocc, nvir)
    return intermed

def initialize_intermediates(orbitals, ranks=[2]):
    ncor = sum(x.shape[1] for x in orbitals["c"])
    nocc = sum(x.shape[1] for x in orbitals["o"])
    nvir = sum(x.shape[1] for x in orbitals["v"])
    nfrz = sum(x.shape[1] for x in orbitals["w"])
    intermed = {"dsd": DirectSumDiis(3, 9), "t1ov": np.zeros((nocc, nvir))}
    for rank in ranks:
        intermed[f"t{rank}"] = mla.read_tensor(f"t{rank}", rank, nocc, nvir)
    return intermed

def initialize_intermediates_SI(orbitals, ranks=[2]):
    noa = orbitals["o"][0].shape[1]
    nob = orbitals["o"][1].shape[1]
    nva = orbitals["v"][0].shape[1]
    nvb = orbitals["v"][1].shape[1]
    intermed = {"dsd": DirectSumDiis(3, 9), "t1_ov_??": np.zeros((noa, nva)), "t1_ov_??": np.zeros((nob, nvb))}
    for rank in ranks:
        for i in range(rank + 1):
            spinstr = "??" * (rank - i) + "??" * i
            shape = [noa] * (rank - i) + [nob] * i + [nva] * (rank - i) + [nvb] * i
            intermed[f"t{rank}_{spinstr}"] = mla.read_tensor_general(f"t{rank}_{spinstr}", shape)
    return intermed

def initialize_intermediates_hf_SI(orbitals, ranks=[2]):
    noa = orbitals["o"][0].shape[1]
    nob = orbitals["o"][1].shape[1]
    nva = orbitals["v"][0].shape[1]
    nvb = orbitals["v"][1].shape[1]
    intermed = {"dsd": DirectSumDiis(3, 9)}
    for rank in ranks:
        for i in range(rank + 1):
            spinstr = "??" * (rank - i) + "??" * i
            shape = [noa] * (rank - i) + [nob] * i + [nva] * (rank - i) + [nvb] * i
            intermed[f"t{rank}_{spinstr}"] = mla.read_tensor_general(f"t{rank}_{spinstr}", shape)
    return intermed

def hermitian_rdm_energy(i):
    sum_en = einsum("ij, ji", i["h_oo"], i["rdm_oo"]) + einsum("ab, ba", i["h_vv"], i["rdm_vv"])
    sum_en += 0.5 * einsum("IJ AB, IJ AB", i["g_oovv"], i["rdm_oovv"])
    sum_en += 0.25 * einsum("IJ KL, IJ KL", i["g_oooo"], i["rdm_oooo"])
    sum_en += 0.25 * einsum("AB CD, AB CD", i["g_vvvv"], i["rdm_vvvv"])
    sum_en += einsum("IA JB, IA JB", i["g_ovov"], i["rdm_ovov"])
    if "rdm_ov" in i:
        sum_en += 2 * einsum("ia, ia", i["h_ov"], i["rdm_ov"])
    if "rdm_ooov" in i:
        sum_en += einsum("IJ KA, IJ KA", i["g_ooov"], i["rdm_ooov"])
        sum_en += einsum("IA BC, IA BC", i["g_ovvv"], i["rdm_ovvv"])
    return sum_en

def hermitian_rdm_energy_SI(i):
    sum_en = einsum("ij, ji", i["h_oo_??"], i["rdm_oo_??"]) + einsum("ab, ba", i["h_vv_??"], i["rdm_vv_??"])
    sum_en += einsum("ij, ji", i["h_oo_??"], i["rdm_oo_??"]) + einsum("ab, ba", i["h_vv_??"], i["rdm_vv_??"])
    sum_en += 0.5 * einsum("IJ AB, IJ AB", i["g_oovv_????"], i["rdm_oovv_????"])
    sum_en += 2 * einsum("IJ AB, IJ AB", i["g_oovv_????"], i["rdm_oovv_????"])
    sum_en += 0.5 * einsum("IJ AB, IJ AB", i["g_oovv_????"], i["rdm_oovv_????"])
    sum_en += 0.25 * einsum("IJ KL, IJ KL", i["g_oooo_????"], i["rdm_oooo_????"])
    sum_en += einsum("IJ KL, IJ KL", i["g_oooo_????"], i["rdm_oooo_????"])
    sum_en += 0.25 * einsum("IJ KL, IJ KL", i["g_oooo_????"], i["rdm_oooo_????"])
    sum_en += 0.25 * einsum("AB CD, AB CD", i["g_vvvv_????"], i["rdm_vvvv_????"])
    sum_en += einsum("AB CD, AB CD", i["g_vvvv_????"], i["rdm_vvvv_????"])
    sum_en += 0.25 * einsum("AB CD, AB CD", i["g_vvvv_????"], i["rdm_vvvv_????"])
    sum_en += einsum("IA JB, IA JB", i["g_ovov_????"], i["rdm_ovov_????"])
    sum_en += einsum("IA JB, IA JB", i["g_ovov_????"], i["rdm_ovov_????"])
    sum_en += 2 * einsum("IA BJ, IA BJ", i["g_ovvo_????"], i["rdm_ovvo_????"])
    sum_en += einsum("AI BJ, AI BJ", i["g_vovo_????"], i["rdm_vovo_????"])
    sum_en += einsum("IA JB, IA JB", i["g_ovov_????"], i["rdm_ovov_????"])
    if "rdm_ooov_????" in i:
        sum_en += einsum("IJ KA, IJ KA", i["g_ooov_????"], i["rdm_ooov_????"])
        sum_en += 2 * einsum("IJ KA, IJ KA", i["g_ooov_????"], i["rdm_ooov_????"])
        sum_en += 2 * einsum("IJ AK, IJ AK", i["g_oovo_????"], i["rdm_oovo_????"])
        sum_en += einsum("IJ KA, IJ KA", i["g_ooov_????"], i["rdm_ooov_????"])
        sum_en += einsum("IA BC, IA BC", i["g_ovvv_????"], i["rdm_ovvv_????"])
        sum_en += 2 * einsum("IA BC, IA BC", i["g_ovvv_????"], i["rdm_ovvv_????"])
        sum_en += 2 * einsum("AI BC, AI BC", i["g_vovv_????"], i["rdm_vovv_????"])
        sum_en += einsum("IA BC, IA BC", i["g_ovvv_????"], i["rdm_ovvv_????"])
    if "rdm_ov_??" in i:
        sum_en += 2 * einsum("ia, ia", i["h_ov_??"], i["rdm_ov_??"]) + 2 * einsum("ia, ia", i["h_ov_??"], i["rdm_ov_??"])
    return sum_en

