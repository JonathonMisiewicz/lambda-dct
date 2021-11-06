import numpy as np
import scipy.misc
from pilot_implementations.multilinear.tensor import einsum
from typing import Callable, Union

# TODO: Get a better finite difference code. Perhaps repurpose Psi's?
def central_difference(f: Callable[np.ndarray, Union[float, np.ndarray]], x: np.ndarray, step: Union[float, np.ndarray], npts: Union[int, None] = None, nder: int = 1) -> np.ndarray:
    """
    Evaluates f'(x) by finite differentiation.

    Input
    -----
    f:
        Takes in a 1d array of floats. Outputs a float or ndarray. The function we're differentiating.
    x:
        1D vector. We evaluate the derive of f at this point.
    step:
        The displacement size. If an array is supplied, component i is the step to displace
        component i of x. If a float is supplied, use it for all components.
    npts:
        The number of points to use in the stencil.
    nder:
        Take the nder'th derivative of this function.

    Output
    ------
        The target derivative. The dimensions are np.shape(x) by np.shape(f(x))
        f(x) may be a multidimensional array or float.
    """
    if npts is None:
        npts = 1 + nder + nder % 2
    if not np.ndim(step):
        step = float(step) * np.ones_like(x)
    weights = scipy.misc.central_diff_weights(Np=npts, ndiv=nder)

    def derivative(index):
        # Construct our displacement
        dx = np.zeros_like(x)
        dx[index] = step[index]
        disp_grid = [np.array(x) + (k-npts//2) * dx for k in range(npts)]
        vals = tuple(map(f, disp_grid))
        return sum(np.array(vals) * weights / (step[index] ** nder))

    der = tuple(map(derivative, np.ndindex(np.shape(x))))
    shape = np.shape(x) + np.shape(f(x))
    return np.reshape(der, shape)


def hellmann_test(solve: Callable, h_ao: np.ndarray, p_ao: np.ndarray, *args, e_thresh: float = 1e-10, r_thresh: float = 1e-8, **kwargs) -> np.ndarray:
    """
    Input
    -----
    solve:
        A "solver" function. Must have a signature compatible with the single_point function returning a float.
    h_ao:
        One-electron spinorbital integrals.
    p_ao:
        Dipole spinorbital integrals. The first index is x, y, z.
    e_thresh:
        The energy convergence to pass to the solver.
    r_thresh:
        The residual convergence to pass to the solver.

    Other arguments supplied as needed by the solver.

    Output
    ------
    np.ndarray
        The dipole moment computed by finite difference.
    """
    def single_point(f):
        hp_ao = h_ao - einsum("x, p q x->p q", f, p_ao)
        # The first argument to solve is the nuclear repulstion energy. This is constant for all displacements,
        # so we don't actually need it.
        return solve(0, hp_ao, *args, e_thresh=e_thresh, r_thresh=r_thresh, **kwargs)[0]["energy"]

    return central_difference(single_point, np.array((0.0, 0.0, 0.0)), 0.002, npts=9)

