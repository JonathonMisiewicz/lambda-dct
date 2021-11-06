from functools import partial

from pilot_implementations.math_util.solvers import spinorbital
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm.taylor import proc

simultaneous = partial(spinorbital.vanilla,
        compute_energy = common.hermitian_rdm_energy,
        compute_step = proc.simultaneous_step,
        ) 
