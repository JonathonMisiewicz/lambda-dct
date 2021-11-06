from sympy.utilities.iterables import multiset_permutations
import numpy as np

def antisymmetrize_axes(tensor: np.ndarray, *axis_pairs: tuple[int, int]) -> np.ndarray:
    """ Apply P-(p/q) to the tensor. """
    for axis_pair in axis_pairs:
        tensor -= tensor.swapaxes(*axis_pair)
    return tensor

def antisymmetrize_axes_plus(tensor: np.ndarray, *axis_data: tuple[tuple[int]]) -> np.ndarray:
    """ Each inner tuple are a group of indices that should be antisymmetric among themselves.
    Each outer tuple is a group of indices that is already antisymmetric among themselves.
    This function handles the remaining antisymmetrizations. """
    returned_tensor = tensor.copy()
    for row in axis_data:
        old_tensor = returned_tensor.copy()
        returned_tensor = np.zeros(returned_tensor.shape)
        integer_mask = []
        permutation_blocks = []
        old_ordering = []
        for i, block in enumerate(row):
            integer_mask += [i] * len(block)
            permutation_blocks.append(block)
            old_ordering += block
        old_ordering = sorted(old_ordering)
        for permutation in multiset_permutations(integer_mask):
            next_index_per_block = [0] * len(permutation_blocks)
            new_ordering = []
            for block, old_axis in zip(permutation, old_ordering):
                next_index = next_index_per_block[block]
                new_axis = permutation_blocks[block][next_index]
                new_ordering.append(new_axis)
                next_index_per_block[block] += 1
            returned_tensor += np.moveaxis(old_tensor, new_ordering, old_ordering) * find_parity(new_ordering, old_ordering)
    return returned_tensor

def find_parity(list1: list, list2: list) -> int:
    """ Find the parity of the permutation between these. """
    num_flips = 0 
    permutation = [list1.index(i) for i in list2]
    num_elts = len(permutation)
    for i in range(num_elts):
        index_of_i = permutation.index(i)
        num_flips += index_of_i
        permutation.pop(index_of_i)
    return (-1) ** num_flips


