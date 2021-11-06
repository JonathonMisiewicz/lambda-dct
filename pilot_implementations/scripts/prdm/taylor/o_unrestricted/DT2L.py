from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D2

def residual(intermed):
    D2.residual(intermed)
    intermed["r3_ααα"] = 0
    intermed["r3_ααβ"] = 0
    intermed["r3_αββ"] = 0
    intermed["r3_βββ"] = 0
    param.T2_cumulant_residual(intermed)
    param.T2_opdm_residual_L(intermed)

def cumulant2(i):
    D2.cumulant2(i)
    param.T2_cumulant(i)

def RDM1(i):
    D2.RDM1(i)
    param.T2_opdm_L(i)

product = D2.product

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, "half")

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_SI(x, [2, 3])
                      )
simultaneous.SI = True

