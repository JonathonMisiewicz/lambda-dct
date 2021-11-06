from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D4T2U

def residual(intermed):
    D4T2U.residual(intermed)
    param.T3_cumulant_residual_U(intermed)
    param.T3_opdm_residual_U(intermed)

def cumulant2(i):
    D4T2U.cumulant2(i)
    param.T3_cumulant_U(i)

def RDM1(i):
    D4T2U.RDM1(i)
    param.T3_opdm_U(i)

product = D4T2U.product

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_SI(x, [2, 3])
                      )
simultaneous.SI = True

