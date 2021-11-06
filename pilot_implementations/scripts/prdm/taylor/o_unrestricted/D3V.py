from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D2

def residual(intermed):
    D2.residual(intermed)
    param.D3_cumulant_residual_V(intermed)

def cumulant2(i):
    D2.cumulant2(i)
    param.D3_cumulant_V(i)

RDM1 = D2.RDM1
product = D2.product

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = common.initialize_intermediates_SI
                      )
simultaneous.SI = True

