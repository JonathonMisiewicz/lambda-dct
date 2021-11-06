from functools import partial

from pilot_implementations.scripts.prdm import common
from pilot_implementations.math_util.solvers import unrestricted
from pilot_implementations.scripts.prdm.taylor import proc

simultaneous = partial(unrestricted.simultaneous,
        compute_energy = common.hermitian_rdm_energy_SI,
        compute_orbital_residual = unrestricted.hermitian_block_orbital_gradient,
        compute_step = proc.simultaneous_step_SI,
        )

simultaneous.SI = True
