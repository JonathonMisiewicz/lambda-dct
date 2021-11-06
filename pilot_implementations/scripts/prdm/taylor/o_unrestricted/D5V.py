from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D4V

def residual(intermed):
    D4V.residual(intermed)
    param.D5_cumulant_residual_V(intermed)

def cumulant2(i):
    D4V.cumulant2(i)
    param.D5_cumulant_V(i)

RDM1 = D4V.RDM1
product = D4V.product

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = common.initialize_intermediates_SI
                      )
simultaneous.SI = True

