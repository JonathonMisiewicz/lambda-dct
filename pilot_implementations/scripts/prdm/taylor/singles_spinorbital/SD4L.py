from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import spinorbital_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed
from . import proc, SD3L

def residual(intermed):
    SD3L.residual(intermed)
    param.D4_cumulant_residual_L(intermed)
    param.D4_opdm_residual_L(intermed)
    param.D4_opdm_product_residual(intermed)
    param.S4_cumulant_residual_L(intermed)
    param.S4_opdm_residual_L(intermed)
    param.S4_opdm_product_residual_L(intermed)

def cumulant2(i):
    SD3L.cumulant2(i)
    param.D4_cumulant_L(i)
    param.S4_cumulant_L(i)

def RDM1(i):
    SD3L.RDM1(i)
    param.D4_opdm_L(i)
    param.S4_opdm_L(i)

def product(i):
    SD3L.product(i)
    param.D4_opdm_product(i)
    param.S4_opdm_product_L(i)

def intermed(i):
    return build_intermed(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf(x, [1, 2])
                      ) 
