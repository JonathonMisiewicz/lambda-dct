from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, D3V

def residual(intermed):
    D3V.residual(intermed)
    param.D4_cumulant_residual_V(intermed)
    param.D4_opdm_residual_V(intermed)
    param.D4_opdm_product_residual(intermed)

def cumulant2(i):
    D3V.cumulant2(i)
    param.D4_cumulant_V(i)

def RDM1(i):
    D3V.RDM1(i)
    param.D4_opdm_V(i)

def product(i):
    D3V.product(i)
    param.D4_opdm_product(i)

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = common.initialize_intermediates_SI
                      )
simultaneous.SI = True

