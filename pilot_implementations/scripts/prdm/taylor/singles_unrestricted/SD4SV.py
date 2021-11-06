from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, SD3SV

def residual(intermed):
    SD3SV.residual(intermed)
    param.D4_cumulant_residual_V(intermed)
    param.D4_opdm_residual_V(intermed)
    param.D4_opdm_product_residual(intermed)
    param.S4_cumulant_residual_SV(intermed)
    param.S4_opdm_residual_SV(intermed)
    param.S4_opdm_product_residual_SV(intermed)

def cumulant2(i):
    SD3SV.cumulant2(i)
    param.D4_cumulant_V(i)
    param.S4_cumulant_SV(i)

def RDM1(i):
    SD3SV.RDM1(i)
    param.D4_opdm_V(i)
    param.S4_opdm_SV(i)

def product(i):
    SD3SV.product(i)
    param.D4_opdm_product(i)
    param.S4_opdm_product_SV(i)

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf_SI(x, [1, 2])
                      )
