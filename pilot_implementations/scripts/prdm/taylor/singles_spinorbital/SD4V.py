from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import spinorbital_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed
from . import proc, SD3V

def residual(intermed):
    SD3V.residual(intermed)
    param.D4_cumulant_residual_V(intermed)
    param.D4_opdm_residual_V(intermed)
    param.D4_opdm_product_residual(intermed)
    param.S4_cumulant_residual_SV(intermed)
    param.S4_cumulant_residual_WV(intermed)
    param.S4_opdm_residual_SV(intermed)
    param.S4_opdm_residual_WV(intermed)
    param.S4_opdm_product_residual_SV(intermed)
    param.S4_opdm_product_residual_WV(intermed)

def cumulant2(i):
    SD3V.cumulant2(i)
    param.D4_cumulant_V(i)
    param.S4_cumulant_SV(i)
    param.S4_cumulant_WV(i)

def RDM1(i):
    SD3V.RDM1(i)
    param.D4_opdm_V(i)
    param.S4_opdm_SV(i)
    param.S4_opdm_WV(i)

def product(i):
    SD3V.product(i)
    param.D4_opdm_product(i)
    param.S4_opdm_product_SV(i)
    param.S4_opdm_product_WV(i)

def intermed(i):
    return build_intermed(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf(x, [1, 2])
                      ) 
