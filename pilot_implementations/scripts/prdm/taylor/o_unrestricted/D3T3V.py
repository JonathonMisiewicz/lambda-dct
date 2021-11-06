from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D3T2V

def residual(intermed):
    D3T2V.residual(intermed)
    param.T3_cumulant_residual_SV(intermed)
    param.T3_cumulant_residual_WV(intermed)
    param.T3_opdm_residual_SV(intermed)
    param.T3_opdm_residual_WV(intermed)

def cumulant2(i):
    D3T2V.cumulant2(i)
    param.T3_cumulant_SV(i)
    param.T3_cumulant_WV(i)

def RDM1(i):
    D3T2V.RDM1(i)
    param.T3_opdm_SV(i)
    param.T3_opdm_WV(i)

product = D3T2V.product

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_SI(x, [2, 3])
                      )
simultaneous.SI = True

