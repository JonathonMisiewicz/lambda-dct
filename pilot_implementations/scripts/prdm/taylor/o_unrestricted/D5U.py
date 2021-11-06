from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D4U

def residual(intermed):
    D4U.residual(intermed)
    param.D5_cumulant_residual_U(intermed)

def cumulant2(i):
    D4U.cumulant2(i)
    param.D5_cumulant_U(i)

RDM1 = D4U.RDM1
product = D4U.product

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = common.initialize_intermediates_SI
                      )
simultaneous.SI = True

