from collections import OrderedDict
from typing import Iterable

import numpy as np
import scipy.linalg as spla

from . import tensor as mla

def orb_rot(intermed: dict, start_orbitals: OrderedDict[str: tuple[np.ndarray, np.ndarray]]) -> OrderedDict[str: tuple[np.ndarray, np.ndarray]]:
    """ Apply a rotation given by the t amplitudes to the orbitals, returning the new orbitals. """
    dim, dim_a = dict(), dict()
    for key, val in start_orbitals.items():
        dim_a[key] = val[0].shape[1]
        dim[key] = val[0].shape[1] + val[1].shape[1]
    # Assemble the blocks into the X matrix which we need to exponentiate
    X = []
    for i, space_i in enumerate(start_orbitals):
        X.append([])
        for j, space_j in enumerate(start_orbitals):
            if i > j and f"t1{space_j}{space_i}" in intermed: # Lower triangle
                block = intermed.get(f"t1{space_j}{space_i}", np.zeros(0)).T
            elif i < j and f"t1{space_i}{space_j}" in intermed: # Upper triangle
                block = intermed.get(f"t1{space_i}{space_j}", np.array(0)) * -1
            else:
                block = np.zeros((dim[space_i], dim[space_j]))
            X[-1].append(block)
    U = spla.expm(np.block(X))
    # Space 1 Alpha, Space 1 Beta, Space 2 Alpha, etc.
    reassembled_C = np.hstack(tuple(i for sub in start_orbitals.values() for i in sub))
    new_C = reassembled_C @ U
    # From our C matrix, construct the new orbitals.
    new_orbitals = OrderedDict()
    for key in start_orbitals:
        alpha_split = val[0]
        # Split off the orbitals of this subspace from the others.
        alpha, beta, new_C = np.hsplit(new_C, [dim_a[key], dim[key]])
        new_orbitals[key] = (alpha, beta)
    return new_orbitals

def orb_rot_SI(intermed: dict, start_orbitals: OrderedDict[str: tuple[np.ndarray, np.ndarray]]) -> OrderedDict[str: tuple[np.ndarray, np.ndarray]]:
    """ Apply a rotation given by the t amplitudes to the orbitals, returning the new orbitals. """
    noa, nva = intermed["t1_ov_α"].shape
    Zoa = np.zeros((noa, noa)) 
    Zva = np.zeros((nva, nva))
    Xa = np.block([[Zoa, -intermed["t1_ov_α"]],
                  [intermed["t1_ov_α"].T, Zva]])
    U = spla.expm(np.block(Xa))
    reassembled_C = np.hstack( tuple(i[0] for i in start_orbitals.values()))
    new_Ca = reassembled_C @ U

    nob, nvb = intermed["t1_ov_β"].shape
    Zob = np.zeros((nob, nob)) 
    Zvb = np.zeros((nvb, nvb))
    Xb = np.block([[Zob, -intermed["t1_ov_β"]],
                  [intermed["t1_ov_β"].T, Zvb]])
    U = spla.expm(np.block(Xb))
    reassembled_C = np.hstack( tuple(i[1] for i in start_orbitals.values()))
    new_Cb = reassembled_C @ U

    new_orbitals = OrderedDict()
    new_Coa, new_Cva = np.hsplit(new_Ca, [noa])
    new_Cob, new_Cvb = np.hsplit(new_Cb, [nob])
    new_orbitals["o"] = (new_Coa, new_Cob)
    new_orbitals["v"] = (new_Cva, new_Cvb)

    return new_orbitals


def antisym_subspace(tensor: np.ndarray, integral_transformers: tuple[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """ Transform each index of the tensor into the corresponding subspaces, and then antisymmetrize.. """
    num_pairs = len(integral_transformers) // 2
    tensor_trans = spatial_subspace(tensor, integral_transformers)
    if num_pairs == 1:
        return tensor_trans 
    elif num_pairs == 2:
        integral_transformers = (integral_transformers[0], integral_transformers[1], integral_transformers[3], integral_transformers[2])
        second_tensor = spatial_subspace(tensor, integral_transformers)
        return tensor_trans - second_tensor.swapaxes(2, 3)
    else:
        print("You have an example of an integral that requires you to")
        print("antisymmetrize more than the simplest case.")
        print("Write a proper antisymmetrizing function to replace this.")
        raise Exception


def request_asym(integral: np.ndarray, space_dict, strings: Iterable[str]) -> dict[str, np.ndarray]:
    """ Construct and return a dictionary with the intrgrals transformed in the various spaces."""
    integral_dict = {}
    for string in strings:
        integral_dict["".join(string)] = antisym_subspace(integral, [space_dict[i] for i in string])
    return integral_dict


def mso_to_aso(tensor: np.ndarray, subspaces: tuple[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """ Given a tensor in spin orbitals and separate matrices specifying an alpha and beta transform,
    perform a transformation into the new composite basis specified by the subspaces - assume a different
    transformation for each index.
    """
    if not tensor.size:
        # The tensor is empty. We need to return a trivial tensor of the correct shape.
        # The first 0 index selects the alpha tuple, and the second gets the row number.
        dims = [2 * subspace[0].shape[0] for subspace in subspaces]
        return np.zeros(dims)

    for i, (alpha, beta) in enumerate(subspaces):
        space = np.vstack([np.hstack((alpha, np.zeros(beta.shape))), np.hstack((np.zeros(alpha.shape), beta))]).T
        
        tensor = mla.one_index_transform(tensor, i, space)

    return tensor

def spatial_subspace(tensor: np.ndarray, subspaces: tuple[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Transform each index of the tensor into the corresponding subspaces.

    tensor: A real tensor with an equal number of upper and lower indices
    subspaces A tuple of tuples with the transformation specifications.
        The innermost tuple specifies an alpha spatial subspace and a beta spatial
        subspace, with AOs are rows and basis vectors as columns.
        The first "n" indices are assumed to be the bras
        and the next "n" are assumed to be kets. Other indices are unchanged.
    """

    ndim = len(tensor.shape)

    num_pairs = len(subspaces) // 2
    bras = subspaces[:num_pairs]
    kets = subspaces[num_pairs:]

    for i, (bra, ket) in enumerate(zip(bras, kets)):
        braspace = np.hstack(bra)
        ketspace = np.hstack(ket)

        bra_axis = i
        ket_axis = i + num_pairs
        # First, perform the spatial transformation. Ignore spin.
        tensor = mla.one_index_transform(tensor, bra_axis, braspace)
        tensor = mla.one_index_transform(tensor, ket_axis, ketspace)
        # Now, perform the spin transformation.
        # All we need to do is zero the blocks where the spins disagree.
        num_bra_a = bra[0].shape[1]
        num_ket_a = ket[0].shape[1]
        # Construct the indices for the opposite spin blocks.
        ab_block = [slice(None)] * ndim
        ba_block = ab_block[:]
        ab_block[bra_axis], ab_block[ket_axis] = slice(None, num_bra_a), slice(num_ket_a, None)
        ba_block[bra_axis], ba_block[ket_axis] = slice(num_bra_a, None), slice(None, num_ket_a)
        # Indices constructed. Now, set the block to zero to end the transform!
        tensor[tuple(ab_block)] = tensor[tuple(ba_block)] = 0
    return tensor


def to_spinorb(tensor: np.ndarray, electron_indices: tuple[tuple[int]]) -> np.ndarray:
    """ Convert a tensor from AO indices to spin orbital indices.

    electron_indices are stored as pairs of indices to simultaneously convert.
    This is required because in non-antisymmetrized integrals, indices corresponding
    to the same electron must have the same spin."""
    for index_pair in electron_indices:
        tensor = _one_electron_transform(tensor, index_pair)
    return tensor


def _one_electron_transform(tensor: np.ndarray, index_pair: tuple[int, int]) -> np.ndarray:
    """ Expand a single "electron" in an integral tensor from being
    in spatial to spin orbital indices.
    The indices in the index pair must correspond to the same electron.
    This function should therefore never be called on antisymmetrized integrals. """

    # Move our two electron indices to the end, for the kron call.
    tensor = np.moveaxis(tensor, index_pair, (-2, -1))
    # Now, convert. All alpha indices precede all beta indices...
    # ...but the spatial ordering is preserved.
    tensor = np.kron(np.eye(2), tensor)
    # Reverse the previous axis transform.
    tensor = np.moveaxis(tensor, (-2, -1), index_pair)
    return tensor


