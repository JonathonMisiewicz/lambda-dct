from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D4T2L

def residual(intermed):
    D4T2L.residual(intermed)
    param.T3_cumulant_residual_L(intermed)

def cumulant2(i):
    D4T2L.cumulant2(i)
    param.T3_cumulant_L(i)

RDM1 = D4T2L.RDM1
product = D4T2L.product

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, "half")

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_SI(x, [2, 3])
                      )
simultaneous.SI = True

