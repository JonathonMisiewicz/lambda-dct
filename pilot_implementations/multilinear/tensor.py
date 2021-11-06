import numpy as np
from opt_einsum import contract

einsum = contract

def contra_transform(tensor: np.ndarray, matrix: np.ndarray, exclude: set[int] =set()) -> np.ndarray:
    """
    Perform a one-index change of basis on all indices of a multi-index tensor, except for
    those indices specified in exclude.
    """
    for axis in range(tensor.ndim):
        if axis not in exclude:
            tensor = one_index_transform(tensor, axis, matrix)
    return tensor


def one_index_transform(tensor: np.ndarray, axis: int, matrix: np.ndarray) -> np.ndarray:
    """
    Perform a one-index change of basis on a specific axis of a multi-index tensor.

    The matrix specifies the change of basis.
    Rows of the matrix index the original basis, and columns index the new one.
    Change the basis for a single vector space in the tensor.
    Can be used to restrict to a subspace of that vector space.
    """
    to_move = np.tensordot(tensor, matrix, (axis, 0))
    return np.moveaxis(to_move, -1, axis)

def broadcaster(multiplicity: int, first_tensor: np.ndarray, second_tensor: np.ndarray) -> np.ndarray:
    """
    Syntactic sugar on the common broadcaster use case where the first _m_ indices need to be broadcasted
    with the same vector, and the other _m_ indices need to be broadcasted with a different vector.
    """
    axis_tuple = [first_tensor] * multiplicity + [second_tensor] * multiplicity
    return full_broadcaster(axis_tuple)

def full_broadcaster(axes: tuple[np.ndarray]) -> np.ndarray:
    """ Create a tensor where element ijkl is elt. i of axis 1 + elt. j of axis 2, etc. """
    tensor = np.zeros(tuple(len(axis) for axis in axes))
    for i, axis in enumerate(axes):
        dim_tuple = tuple((-1 if i == j else 1) for j in range(len(axes)))
        tensor += axis.reshape(dim_tuple) # Reshape axis for easy broadcasting into final target
    return tensor

def transform_all_indices(tensor: np.ndarray, matrix_tuple: tuple[np.ndarray]) -> np.ndarray:
    """ Transform the i'th index of tensor with the i'th matrix in the tuple. """
    assert len(matrix_tuple) == len(tensor.shape)
    for i, matrix in enumerate(matrix_tuple):
        tensor = one_index_transform(tensor, i, matrix)
    return tensor

def compute_basis_transformation(from_basis: np.ndarray, target_basis: np.ndarray) -> np.ndarray:
    """
    Given two "matrices" of basis vectors for a space, construct the matrix U where Umn is the coefficient
    of basis vector m (of from_basis") in basis vector n (of "target_basis").
    target_basis = from_basis @ U; each basis vector is a column, and each row vector is some other basis
    """
    return np.linalg.inv(from_basis) @ target_basis


def read_tensor(name, num_indices, nina, nvir, failfast = False) -> np.ndarray:
    """
    Syntactic sugar on the common read_tensor use case where the first _m_ indices have the same shape
    and the other _m_ indices have a different shape.
    """
    return read_tensor_general(name, (nina,) * num_indices + (nvir,) * num_indices, failfast)

def read_tensor_general(name: str, shape: tuple[int], failfast: bool = False) -> np.ndarray:
    """ Attempt to read a tensor from disk. Default to zeroing, but override with failfast.

    namme is the tensor name, and shape specifies the expected shape of the tensor."""
    try:
        tensor = np.load(name + ".npy")
        if tensor.shape == tuple(shape): return tensor
    except IOError:
        pass
    else:
        if failfast: raise AssertionError
    return np.zeros(shape)

