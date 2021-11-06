from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D4T2SV

def residual(intermed):
    D4T2SV.residual(intermed)
    param.T3_cumulant_residual_SV(intermed)
    param.T3_opdm_residual_SV(intermed)

def cumulant2(i):
    D4T2SV.cumulant2(i)
    param.T3_cumulant_SV(i)

def RDM1(i):
    D4T2SV.RDM1(i)
    param.T3_opdm_SV(i)

product = D4T2SV.product

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_SI(x, [2, 3])
                      )
simultaneous.SI = True

